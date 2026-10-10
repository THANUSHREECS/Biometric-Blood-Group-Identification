import os
from flask import Blueprint, render_template, request, session, redirect, url_for, flash
from src.mfs100.scanner import MFS100Scanner
from scripts.append_ground_truth import append_row

# Blueprint definition
ground_truth_bp = Blueprint('ground_truth', __name__)

@ground_truth_bp.route('/')
def start_page():
    """Render the initial form for a new subject."""
    return render_template('ground_truth.html')

@ground_truth_bp.route('/start', methods=['POST'])
def start_subject():
    """Validate form data and initialise session for a subject."""
    subject_id = request.form.get('subject_id', '').strip()
    blood_group = request.form.get('blood_group', '').strip()
    source = request.form.get('source', '').strip()
    notes = request.form.get('notes', '').strip()

    # Basic validation
    if not subject_id or not blood_group or not source:
        flash('All required fields must be filled.', 'error')
        return redirect(url_for('ground_truth.start_page'))
    if not subject_id.startswith('S') or not subject_id[1:].isdigit() or len(subject_id) != 4:
        flash('Subject ID must be in the form S### (e.g., S001).', 'error')
        return redirect(url_for('ground_truth.start_page'))
    allowed_groups = {'A+', 'A-', 'AB+', 'AB-', 'B+', 'B-', 'O+', 'O-'}
    if blood_group not in allowed_groups:
        flash('Invalid blood group selected.', 'error')
        return redirect(url_for('ground_truth.start_page'))

    # Store in session
    session['subject_id'] = subject_id
    session['blood_group'] = blood_group
    session['confirmation_source'] = source
    session['notes'] = notes
    session['capture_count'] = 0
    session['captures'] = []

    return redirect(url_for('ground_truth.capture_page'))

@ground_truth_bp.route('/capture')
def capture_page():
    """Show capture buttons and current progress."""
    if 'subject_id' not in session:
        return redirect(url_for('ground_truth.start_page'))
    return render_template('ground_truth_capture.html', capture_count=session.get('capture_count', 0))

@ground_truth_bp.route('/capture', methods=['POST'])
def capture_sample():
    """Perform a single fingerprint capture, save BMP, and append CSV row."""
    if 'subject_id' not in session:
        return redirect(url_for('ground_truth.start_page'))

    from app import scanner  # Import scanner instance from main app
    captured_path, _, meta = scanner.capture_with_meta()
    filename = os.path.basename(captured_path)
    nfiq = meta.get('nfiq', '')
    quality = meta.get('quality', '')

    row = {
        'subject_id': session['subject_id'],
        'capture_filename': filename,
        'confirmed_blood_group': session['blood_group'],
        'finger_used': 'LEFT_THUMB',
        'nfiq': str(nfiq),
        'quality': str(quality),
        'confirmation_source': session['confirmation_source'],
        'notes': session['notes']
    }
    try:
        append_row(row)
    except Exception as e:
        flash(f'Error appending CSV: {e}', 'error')
        return redirect(url_for('ground_truth.capture_page'))

    session['capture_count'] = session.get('capture_count', 0) + 1
    captures = session.get('captures', [])
    captures.append({'filename': filename, 'nfiq': nfiq, 'quality': quality})
    session['captures'] = captures
    flash(f'Capture {session["capture_count"]} saved as {filename} (NFIQ={nfiq}, Quality={quality}).', 'info')
    return redirect(url_for('ground_truth.capture_page'))

@ground_truth_bp.route('/finish', methods=['POST'])
def finish_subject():
    """Validate three captures, clear session, and show summary with validation output."""
    if session.get('capture_count', 0) != 3:
        flash('Exactly three captures are required before finishing.', 'error')
        return redirect(url_for('ground_truth.capture_page'))

    import subprocess, sys, os
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    result = subprocess.run([sys.executable, os.path.join(project_root, 'scripts', 'validate_ground_truth.py')], cwd=project_root, capture_output=True, text=True)
    validation_output = result.stdout

    summary = {
        'subject_id': session['subject_id'],
        'blood_group': session['blood_group'],
        'capture_count': session['capture_count'],
        'captures': session['captures'],
        'validation_output': validation_output
    }

    session.clear()
    return render_template('ground_truth_summary.html', summary=summary)
