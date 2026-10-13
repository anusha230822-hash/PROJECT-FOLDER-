import json
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import Count
from django.http import Http404

from .models import (
    UserProfile, SatelliteImage, ClassificationResult,
    DatasetRecord, OntologyConcept
)
from .forms import (
    UserRegistrationForm, UserLoginForm,
    SatelliteImageUploadForm, DatasetRecordForm
)
from .preprocessing import preprocess_image
from .feature_extraction import extract_features
from .ml_classifier import (
    predict_image, is_model_trained, ModelNotTrainedError,
    get_models_dir
)
from .ontology import (
    load_ontology, get_all_concepts_tree, search_ontology,
    get_concept, get_parent_class, get_children, get_relationships
)
from .ontology_mapping import map_prediction_to_ontology
from .visualization import (
    plot_prediction_probabilities, plot_dataset_distribution,
    plot_confusion_matrix
)
from .evaluation import get_saved_evaluation_results
from .utils import (
    get_dataset_statistics, populate_initial_ontology_concepts_in_db,
    ALL_CLASSES
)


# Helper: Check if user is admin
def is_admin_user(user):
    return user.is_authenticated and (user.is_superuser or (hasattr(user, 'profile') and user.profile.role == 'Admin'))


# 1. Home Landing Page
def home_view(request):
    model_ready = is_model_trained()
    stats = get_dataset_statistics()
    total_classifications = ClassificationResult.objects.count()
    return render(request, 'classifier/home.html', {
        'model_ready': model_ready,
        'stats': stats,
        'total_classifications': total_classifications,
    })


# 2. Registration View
def register_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            messages.success(request, "Account created successfully! Please log in.")
            return redirect('login')
    else:
        form = UserRegistrationForm()

    return render(request, 'classifier/register.html', {'form': form})


# 3. Login View
def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f"Welcome back, {user.username}!")
                next_url = request.GET.get('next')
                if next_url:
                    return redirect(next_url)
                if user.is_superuser or (hasattr(user, 'profile') and user.profile.role == 'Admin'):
                    return redirect('admin_dashboard')
                return redirect('dashboard')
            else:
                messages.error(request, "Invalid username or password.")
    else:
        form = UserLoginForm()

    return render(request, 'classifier/login.html', {'form': form})


# 4. Logout View
def logout_view(request):
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('home')


# 5. User Dashboard
@login_required
def dashboard_view(request):
    user_images = SatelliteImage.objects.filter(user=request.user)
    total_uploads = user_images.count()
    completed_classifications = ClassificationResult.objects.filter(satellite_image__user=request.user)
    total_completed = completed_classifications.count()

    recent_results = completed_classifications.select_related('satellite_image').order_by('-created_at')[:5]
    latest_result = recent_results.first() if recent_results.exists() else None

    model_ready = is_model_trained()

    return render(request, 'classifier/dashboard.html', {
        'total_uploads': total_uploads,
        'total_completed': total_completed,
        'recent_results': recent_results,
        'latest_result': latest_result,
        'model_ready': model_ready,
    })


# 6. Image Upload & Classification Processing
@login_required
def upload_view(request):
    model_ready = is_model_trained()

    if request.method == 'POST':
        form = SatelliteImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            sat_img = form.save(commit=False)
            sat_img.user = request.user
            sat_img.image_name = request.FILES['image'].name
            sat_img.processing_status = 'Processing'
            sat_img.save()

            # Process the image
            try:
                # Preprocess image & extract features
                processed_rgb, _ = preprocess_image(sat_img.image.path)
                feat_vec, feat_dict = extract_features(processed_rgb)

                if model_ready:
                    # ML Inference
                    pred_class, confidence, probabilities, _, _ = predict_image(sat_img.image.path)
                    
                    # Ontology mapping
                    onto_mapping = map_prediction_to_ontology(pred_class)

                    # Store Knowledge Base record
                    res = ClassificationResult.objects.create(
                        satellite_image=sat_img,
                        predicted_class=pred_class,
                        confidence=confidence,
                        ontology_class=onto_mapping['ontology_class'],
                        ontology_parent=onto_mapping['parent_class'],
                        ontology_root=onto_mapping['root_class'],
                        features_json=json.dumps(feat_dict),
                        probabilities_json=json.dumps(probabilities)
                    )

                    sat_img.processing_status = 'Completed'
                    sat_img.save()
                    messages.success(request, f"Image successfully classified as {pred_class}!")
                    return redirect('result', pk=sat_img.pk)
                else:
                    # DEMO MODE: Model is not trained yet
                    sat_img.processing_status = 'Failed'
                    sat_img.error_message = (
                        "Model is not trained yet. Please add the dataset and run the training script."
                    )
                    sat_img.save()
                    messages.warning(
                        request,
                        "Model is not trained yet. Please add the dataset and run the training script (scripts/train_model.py)."
                    )
                    # Redirect to result page in DEMO inspection mode with extracted features
                    return redirect('result_demo', pk=sat_img.pk)

            except Exception as e:
                sat_img.processing_status = 'Failed'
                sat_img.error_message = str(e)
                sat_img.save()
                messages.error(request, f"Processing failed: {str(e)}")
                return redirect('upload')
    else:
        form = SatelliteImageUploadForm()

    return render(request, 'classifier/upload.html', {
        'form': form,
        'model_ready': model_ready,
    })


# 7. Result View (Trained Model)
@login_required
def result_view(request, pk):
    sat_img = get_object_or_404(SatelliteImage, pk=pk)

    # Permission check: owner or admin
    if sat_img.user != request.user and not is_admin_user(request.user):
        messages.error(request, "You are not authorized to view this result.")
        return redirect('dashboard')

    if not hasattr(sat_img, 'result'):
        # Fallback to demo view if no result record exists
        return redirect('result_demo', pk=pk)

    result = sat_img.result

    # Parse features
    features = {}
    if result.features_json:
        try:
            features = json.loads(result.features_json)
        except Exception:
            features = {}

    # Parse probabilities and generate chart
    probabilities = {}
    chart_base64 = None
    if result.probabilities_json:
        try:
            probabilities = json.loads(result.probabilities_json)
            chart_base64 = plot_prediction_probabilities(probabilities)
        except Exception:
            pass

    # Ontology info
    onto_info = map_prediction_to_ontology(result.predicted_class)

    return render(request, 'classifier/result.html', {
        'sat_img': sat_img,
        'result': result,
        'features': features,
        'probabilities': probabilities,
        'chart_base64': chart_base64,
        'onto_info': onto_info,
        'is_demo': False,
    })


# 8. Demo Result View (When model not trained yet, shows extracted features)
@login_required
def result_demo_view(request, pk):
    sat_img = get_object_or_404(SatelliteImage, pk=pk)

    if sat_img.user != request.user and not is_admin_user(request.user):
        messages.error(request, "You are not authorized to view this image.")
        return redirect('dashboard')

    features = {}
    try:
        processed_rgb, _ = preprocess_image(sat_img.image.path)
        _, features = extract_features(processed_rgb)
    except Exception as e:
        features = {'error': str(e)}

    return render(request, 'classifier/result.html', {
        'sat_img': sat_img,
        'result': None,
        'features': features,
        'is_demo': True,
        'demo_message': "Model is not trained yet. Please add the dataset and run the training script."
    })


# 9. Classification History View
@login_required
def history_view(request):
    if is_admin_user(request.user):
        results_qs = ClassificationResult.objects.select_related('satellite_image', 'satellite_image__user')
    else:
        results_qs = ClassificationResult.objects.filter(
            satellite_image__user=request.user
        ).select_related('satellite_image')

    # Filtering by class
    selected_class = request.GET.get('class', '')
    if selected_class and selected_class != 'All':
        results_qs = results_qs.filter(predicted_class=selected_class)

    # Search by image name
    search_q = request.GET.get('q', '').strip()
    if search_q:
        results_qs = results_qs.filter(satellite_image__image_name__icontains=search_q)

    paginator = Paginator(results_qs, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    return render(request, 'classifier/history.html', {
        'page_obj': page_obj,
        'selected_class': selected_class,
        'search_q': search_q,
        'all_classes': ALL_CLASSES,
        'is_admin': is_admin_user(request.user),
    })


# 10. Ontology Explorer View
def ontology_view(request):
    populate_initial_ontology_concepts_in_db()
    try:
        onto_tree = get_all_concepts_tree()
        onto_classes_count = len(list(load_ontology().classes()))
    except Exception as e:
        onto_tree = {}
        onto_classes_count = 0
        messages.error(request, f"Error accessing ontology file: {e}")

    db_concepts = OntologyConcept.objects.all()

    return render(request, 'classifier/ontology.html', {
        'onto_tree': onto_tree,
        'onto_classes_count': onto_classes_count,
        'db_concepts': db_concepts,
    })


# 11. Ontology Search View
def search_view(request):
    query = request.GET.get('q', '').strip()
    results = []

    if query:
        try:
            results = search_ontology(query)
            if not results:
                # Also search DB concepts
                db_matches = OntologyConcept.objects.filter(concept_name__icontains=query)
                for item in db_matches:
                    results.append({
                        'name': item.concept_name,
                        'label': item.concept_name,
                        'description': item.description or 'Semantic concept registered in knowledge-base.',
                        'parent': item.parent_concept or 'None',
                        'parent_label': item.parent_concept or 'None',
                        'children': [],
                        'relationships': [{'source': item.concept_name, 'relationship': item.relationship, 'target': item.parent_concept}] if item.parent_concept else []
                    })
        except Exception as e:
            messages.error(request, f"Search error: {str(e)}")

    return render(request, 'classifier/search.html', {
        'query': query,
        'results': results,
    })


# 12. Model Performance Evaluation View
def performance_view(request):
    eval_results = get_saved_evaluation_results()
    cm_chart_base64 = None

    if eval_results and 'confusion_matrix' in eval_results and 'classes' in eval_results:
        cm_chart_base64 = plot_confusion_matrix(
            eval_results['confusion_matrix'],
            eval_results['classes']
        )

    stats = get_dataset_statistics()
    dist_chart_base64 = plot_dataset_distribution(stats['train'])

    return render(request, 'classifier/performance.html', {
        'eval_results': eval_results,
        'cm_chart_base64': cm_chart_base64,
        'dist_chart_base64': dist_chart_base64,
        'stats': stats,
        'model_ready': is_model_trained(),
    })


# 13. Admin Dashboard View
@login_required
@user_passes_test(is_admin_user)
def admin_dashboard_view(request):
    populate_initial_ontology_concepts_in_db()

    total_users = UserProfile.objects.count()
    total_images = SatelliteImage.objects.count()
    total_results = ClassificationResult.objects.count()
    dataset_stats = get_dataset_statistics()
    eval_results = get_saved_evaluation_results()

    recent_uploads = SatelliteImage.objects.select_related('user').order_by('-uploaded_at')[:8]
    class_distribution = ClassificationResult.objects.values('predicted_class').annotate(count=Count('predicted_class'))

    dist_chart_base64 = plot_dataset_distribution(dataset_stats['train'])

    return render(request, 'classifier/admin_dashboard.html', {
        'total_users': total_users,
        'total_images': total_images,
        'total_results': total_results,
        'dataset_stats': dataset_stats,
        'eval_results': eval_results,
        'recent_uploads': recent_uploads,
        'class_distribution': class_distribution,
        'dist_chart_base64': dist_chart_base64,
        'model_ready': is_model_trained(),
    })


# 14. Admin Dataset Records Manager
@login_required
@user_passes_test(is_admin_user)
def dataset_manager_view(request):
    if request.method == 'POST':
        form = DatasetRecordForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, "Dataset record uploaded successfully.")
            return redirect('dataset_manager')
    else:
        form = DatasetRecordForm()

    records = DatasetRecord.objects.all()
    paginator = Paginator(records, 12)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    stats = get_dataset_statistics()

    return render(request, 'classifier/dataset_manager.html', {
        'form': form,
        'page_obj': page_obj,
        'stats': stats,
        'classes': ALL_CLASSES,
    })
