# ==============================================================================
# IMPORT LIBRARY
# ==============================================================================
import streamlit as st
import os
import pandas as pd
import json
import numpy as np
import librosa
import noisereduce as nr
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import io
from collections import Counter, defaultdict
import re
import openpyxl

# ==============================================================================
# KONFIGURASI APLIKASI STREAMLIT
# ==============================================================================
st.set_page_config(
    layout="wide",
    page_title="Harmoni Nusantara - Soal Generator"
)

# ==============================================================================
# KONSTANTA GLOBAL
# ==============================================================================
# Pengaturan Audio Processing
SR = 22050
N_FFT = 1024
HOP_LENGTH = 512
N_MFCC = 40
MAX_PAD_LEN = 432
N_SPECTRAL_CONTRAST = 7
N_CHROMA = 12
TOTAL_FEATURES = N_MFCC + N_SPECTRAL_CONTRAST + N_CHROMA
TARGET_SEGMENT_DURATION_SECONDS = 10

# Kelas Klasifikasi Gamelan
GAMELAN_CLASSES = ['angklung', 'baleganjur', 'gong_gede', 'gong_kebyar', 'semar_pegulingan']


# ==============================================================================
# FUNGSI-FUNGSI HELPER
# ==============================================================================
def parse_excel_questions(file_bytes):
    """Membaca file Excel dengan format baru (tanpa kolom provinsi)."""
    parsed_questions = []
    try:
        workbook = openpyxl.load_workbook(io.BytesIO(file_bytes))
        sheet = workbook.active
        headers = [cell.value for cell in sheet[1]]
        
        expected_headers = [
            "Nomor_Level", "ID_Soal_Dalam_Level", "Teks_Pertanyaan", 
            "Opsi_A", "Opsi_B", "Opsi_C", "Opsi_D", "Kunci_Jawaban_Dokumen",
            "Nama_File_Gambar", "Nama_File_Audio_Untuk_Soal"
        ]
        
        missing_headers = [eh for eh in expected_headers[:7] if eh not in headers]
        if missing_headers:
            st.error(f"Header Excel tidak ditemukan: {', '.join(missing_headers)}.")
            return []

        for row_num, row_values in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
            if not any(row_values): continue
            row_data = dict(zip(headers, row_values))
            try:
                level_num = int(row_data.get("Nomor_Level"))
                q_id_in_level = int(row_data.get("ID_Soal_Dalam_Level"))
                teks_soal = str(row_data.get("Teks_Pertanyaan", "")).strip()
                kunci = str(row_data.get("Kunci_Jawaban_Dokumen", "")).strip().lower()
                
                if not teks_soal or not kunci: continue
                if kunci not in ['a', 'b', 'c', 'd']: continue
                
                parsed_questions.append({
                    'level_number': level_num,
                    'id_in_level': q_id_in_level,
                    'teks_soal': teks_soal,
                    'opsi_jawaban': {
                        'a': str(row_data.get("Opsi_A", "")).strip(), 'b': str(row_data.get("Opsi_B", "")).strip(),
                        'c': str(row_data.get("Opsi_C", "")).strip(), 'd': str(row_data.get("Opsi_D", "")).strip()
                    },
                    'kunci_jawaban_dokumen': kunci,
                    'nama_file_gambar': str(row_data.get("Nama_File_Gambar", "")).strip() or None,
                    'nama_file_audio_excel': str(row_data.get("Nama_File_Audio_Untuk_Soal", "")).strip() or None
                })
            except (ValueError, TypeError): st.caption(f"Baris Excel {row_num}: Error konversi data, dilewati.")
    except Exception as e: st.error(f"Gagal memproses file Excel: {e}")
    return parsed_questions

def denoise_audio(y, sr):
    """
    Menghilangkan noise dari sinyal audio menggunakan library noisereduce.
    Noise diestimasi dari bagian audio dengan energi terendah.
    """
    try:
        # Tentukan bagian noise berdasarkan energi terendah
        energy = librosa.feature.rms(y=y)
        if energy.size == 0: return y
        percentile_energy = np.percentile(energy, 10)
        noise_frames_indices = np.where(energy.flatten() < percentile_energy)[0]
        
        # Default noise part jika tidak ada frame noise yang terdeteksi
        noise_part = y[:int(0.1 * len(y))]
        
        if len(noise_frames_indices) > 0:
            noise_samples_indices = librosa.frames_to_samples(noise_frames_indices)
            noise_samples_indices = noise_samples_indices[noise_samples_indices < len(y)]
            if len(noise_samples_indices) > 0:
                noise_part_candidate = y[noise_samples_indices]
                if len(noise_part_candidate) > 0:
                    noise_part = noise_part_candidate

        if len(noise_part) == 0 and len(y) > 0:
            noise_part = y[:int(0.1 * len(y))]
        elif len(y) == 0:
            return y

        # Parameter untuk noise reduction
        GAMELAN_NR_PARAMS = {
            'stationary': False, 'prop_decrease': 0.5, 'n_fft': 1024,
            'win_length': 256, 'use_tqdm': False
        }
        
        # Lakukan noise reduction
        if len(noise_part) > 0:
            return nr.reduce_noise(y=y, y_noise=noise_part, sr=sr, **GAMELAN_NR_PARAMS)
        return y
    except Exception:
        return y

def extract_features_for_streamlit(y, sr):
    """
    Mengekstrak fitur dari sinyal audio untuk input model klasifikasi.
    Termasuk denoising, ekstraksi MFCC, Spectral Contrast, Chroma, normalisasi, dan padding.
    """
    try:
        y_clean = denoise_audio(y, sr)
        if y_clean is None or len(y_clean) == 0:
            return None

        # Ekstraksi fitur
        mfcc = librosa.feature.mfcc(y=y_clean, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=HOP_LENGTH)
        spectral_contrast = librosa.feature.spectral_contrast(y=y_clean, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        chroma = librosa.feature.chroma_cqt(y=y_clean, sr=sr, hop_length=HOP_LENGTH)

        # Gabungkan semua fitur
        features = np.vstack([mfcc, spectral_contrast, chroma])
        if features.shape[0] != TOTAL_FEATURES:
            return None

        # Normalisasi fitur
        mean, std = np.mean(features, axis=1, keepdims=True), np.std(features, axis=1, keepdims=True)
        features = (features - mean) / (std + 1e-6)

        # Padding atau pemotongan agar panjangnya sesuai
        if features.shape[1] < MAX_PAD_LEN:
            features = np.tile(features, int(np.ceil(MAX_PAD_LEN / features.shape[1])))[:, :MAX_PAD_LEN]
        else:
            features = features[:, :MAX_PAD_LEN]

        # Ubah bentuk array sesuai input model
        return features.T.reshape(MAX_PAD_LEN, TOTAL_FEATURES, 1, 1)
    except Exception:
        return None

@st.cache_resource
def load_gamelan_model():
    """
    Memuat model klasifikasi Gamelan dari file .keras.
    Menggunakan cache Streamlit untuk mencegah pemuatan berulang.
    """
    try:
        model_path = 'ConvLSTM_10s.keras'
        if not os.path.exists(model_path):
            st.error(f"File model '{model_path}' tidak ditemukan.")
            return None
        return tf.keras.models.load_model(model_path)
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

def classify_audio_files_streamlit(audio_file_infos, progress_placeholder=None):
    """
    Mengklasifikasikan daftar file audio yang diunggah.
    Setiap file audio dipotong menjadi segmen-segmen, lalu setiap segmen diklasifikasikan.
    Hasil akhir adalah kelas yang paling sering muncul dari semua segmen.
    """
    results = []
    if not model:
        st.error("Model klasifikasi tidak berhasil dimuat.")
        return results

    total = len(audio_file_infos)
    if total == 0:
        return results

    seg_len_samples = SR * TARGET_SEGMENT_DURATION_SECONDS
    for i, f_info in enumerate(audio_file_infos):
        fname, abytes, fid = f_info['name'], f_info['bytes'], f_info['id']
        if progress_placeholder:
            progress_placeholder.progress(i / total, text=f"Proses: {fname} ({i+1}/{total})")

        try:
            # Muat dan resample audio
            y, sr_orig = librosa.load(io.BytesIO(abytes), sr=None)
            if sr_orig != SR:
                y = librosa.resample(y, orig_sr=sr_orig, target_sr=SR)

            # Proses per segmen
            num_samples, seg_preds, pos = len(y), [], 0
            while pos < num_samples:
                seg_y = y[pos:min(pos + seg_len_samples, num_samples)]
                pos += seg_len_samples
                
                # Lewati segmen yang terlalu pendek
                if len(seg_y) < SR * 0.5:
                    if num_samples <= seg_len_samples and len(seg_y) > 0:
                        pass
                    elif len(seg_preds) > 0 or len(seg_y) == 0:
                        continue
                
                # Ekstrak fitur dan prediksi
                feat_seg = extract_features_for_streamlit(seg_y, SR)
                if feat_seg is not None:
                    pred_probs = model.predict(np.expand_dims(feat_seg, axis=0), verbose=0)[0]
                    seg_preds.append(encoder.classes_[np.argmax(pred_probs)])

            # Agregasi hasil dari semua segmen
            if seg_preds:
                counts = Counter(seg_preds)
                probs = {cls: float(counts.get(cls, 0)) / len(seg_preds) for cls in GAMELAN_CLASSES}
                results.append({'filename': fname, 'probabilities': probs, 'id': fid})
            else:
                results.append({'filename': fname, 'probabilities': {}, 'error': 'Tidak ada segmen audio yang valid.', 'id': fid})
        except Exception as e:
            results.append({'filename': fname, 'probabilities': {}, 'error': str(e), 'id': fid})

    if progress_placeholder:
        progress_placeholder.progress(1.0, text=f"Validasi {total} file selesai!")
    return results


# ==============================================================================
# INISIALISASI MODEL, ENCODER, DAN SESSION STATE
# ==============================================================================

# Muat model dan siapkan encoder
model = load_gamelan_model()
encoder = LabelEncoder()
encoder.fit(GAMELAN_CLASSES)

# Inisialisasi session state untuk menyimpan data antar-interaksi
default_session_state = {
    'uploaded_files_info': [],
    'classification_results': [],
    'show_export_dialog': False,
    'uploader_key_counter': 0,
    'uploader_doc_key_counter': 0,
    'show_validation_modal': False,
    'files_being_validated': [],
    'uploaded_question_document_file': None,
    'parsed_document_questions': [],
    'document_parse_error': None,
    'edited_audio_answer_keys': {},
    'audio_to_soal_info_map': {}
}
for key, default_val in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


# ==============================================================================
# FUNGSI-FUNGSI AKSI (CALLBACK)
# ==============================================================================

def open_export_dialog():
    """Membuka dialog ekspor JSON jika file soal sudah diunggah."""
    if not st.session_state.parsed_document_questions:
        st.warning("Unggah dan proses file soal Excel terlebih dahulu.")
    else:
        st.session_state.show_export_dialog = True

def delete_audio_file(file_id):
    """Menghapus file audio dari session state berdasarkan ID-nya."""
    st.session_state.uploaded_files_info = [f for f in st.session_state.uploaded_files_info if f['id'] != file_id]
    st.session_state.classification_results = [r for r in st.session_state.classification_results if r.get('id') != file_id]
    st.session_state.uploader_key_counter += 1
    st.toast("File audio dan hasilnya telah dihapus.", icon="🗑️")

def clear_all_data():
    """Menghapus semua data dari session state untuk memulai dari awal."""
    for k_list in ['uploaded_files_info', 'classification_results', 'parsed_document_questions', 'audio_to_soal_info_map']:
        st.session_state[k_list] = [] if isinstance(st.session_state.get(k_list), list) else {}
    st.session_state.edited_audio_answer_keys = {}
    for k_none in ['uploaded_question_document_file', 'document_parse_error']:
        st.session_state[k_none] = None
    st.session_state.uploader_key_counter += 1
    st.session_state.uploader_doc_key_counter += 1
    st.toast("Semua data telah dihapus.", icon="♻️")

def process_document(doc_file):
    """Memproses file Excel yang diunggah."""
    st.session_state.parsed_document_questions = []
    st.session_state.document_parse_error = None
    st.session_state.edited_audio_answer_keys = {}
    st.session_state.audio_to_soal_info_map = {}
    st.session_state.uploaded_question_document_file = {"name": doc_file.name, "id": doc_file.file_id}
    
    # Validasi ekstensi file
    ext = os.path.splitext(doc_file.name)[1].lower()
    if ext in [".xlsx", ".xls"]:
        parsed = parse_excel_questions(doc_file.getvalue())
        if parsed:
            st.session_state.parsed_document_questions = parsed
            audio_map = {
                q['nama_file_audio_excel']: {'id': q['id_in_level'], 'level': q['level_number']} 
                for q in parsed if q.get('nama_file_audio_excel')
            }
            st.session_state.audio_to_soal_info_map = audio_map
            st.toast(f"Ditemukan {len(audio_map)} soal dengan audio terkait di Excel.")
        elif not st.session_state.document_parse_error:
            st.warning(f"Tidak ada soal yang berhasil diekstrak dari '{doc_file.name}'.")
    else:
        st.session_state.document_parse_error = f"Format file {ext} tidak didukung. Harap gunakan file Excel (.xlsx)."


# ==============================================================================
# TATA LETAK (LAYOUT) APLIKASI STREAMLIT
# ==============================================================================

# --- Judul Aplikasi ---
st.markdown("<h1 style='text-align: center;'>Gamelan Classification & Soal Generator</h1>", unsafe_allow_html=True)
st.markdown("---")

st.markdown("<h4 style='text-align: center;'>1. Unggah File Soal Excel (.xlsx)</h4>", unsafe_allow_html=True)
st.caption("Pastikan Excel memiliki kolom: Nomor_Level, ID_Soal_Dalam_Level, Teks_Pertanyaan, Opsi_A, Opsi_B, Opsi_C, Opsi_D, Kunci_Jawaban_Dokumen, Nama_File_Gambar (ops), Nama_File_Audio_Untuk_Soal (ops).")
uploaded_doc = st.file_uploader(
    "Unggah file Excel",
    type=["xlsx", "xls"],
    key=f"doc_uploader_{st.session_state.uploader_doc_key_counter}"
)

if uploaded_doc:
    current_doc_id = st.session_state.uploaded_question_document_file['id'] if st.session_state.uploaded_question_document_file else None
    if uploaded_doc.file_id != current_doc_id:
        with st.spinner(f"Memproses '{uploaded_doc.name}'..."):
            process_document(uploaded_doc)
        st.rerun()

if st.session_state.uploaded_question_document_file:
    if st.session_state.parsed_document_questions:
        st.info(f"Berhasil memuat **{len(st.session_state.parsed_document_questions)}** soal dari dokumen.")
        # with st.expander("Lihat Pratinjau 3 Soal Pertama dari Dokumen"):
        #     st.json(st.session_state.parsed_document_questions[:3])
    elif st.session_state.document_parse_error:
        st.error(f"Error Parsing Dokumen: {st.session_state.document_parse_error}")

st.markdown("---")

# --- Bagian Unggah File Audio ---
st.markdown("<h4 style='text-align: center;'>Unggah File Audio (WAV/MP3/OGG)</h4>", unsafe_allow_html=True)
st.caption("Jika soal memerlukan audio, pastikan nama file audio yang diunggah sesuai dengan kolom 'Nama_File_Audio_Untuk_Soal' di Excel.")

uploaded_audios = st.file_uploader(
    "Unggah file audio",
    type=["wav", "mp3", "ogg"],
    accept_multiple_files=True,
    key=f"audio_uploader_{st.session_state.uploader_key_counter}"
)

# Tambahkan file audio baru ke session state
if uploaded_audios:
    existing_ids = {f['id'] for f in st.session_state.uploaded_files_info}
    added = False
    soal_info_map = st.session_state.get('audio_to_soal_info_map', {})
    for audio_f in uploaded_audios:
        if audio_f.file_id not in existing_ids:
            st.session_state.uploaded_files_info.append({
                "name": audio_f.name,
                "bytes": audio_f.getvalue(),
                "id": audio_f.file_id,
                "soal_info": soal_info_map.get(audio_f.name)
            })
            added = True
    if added:
        st.rerun()

st.markdown("---")

# --- Tampilan Daftar Audio dan Hasil Validasi ---
col_list_audio, col_hasil_validasi = st.columns(2)

with col_list_audio:
    st.markdown("###### Daftar File Audio Diunggah")
    with st.container(border=True, height=500):
        if not st.session_state.uploaded_files_info:
            st.info("Belum ada file audio yang diunggah.")
        else:
            sorted_files = sorted(
                st.session_state.uploaded_files_info,
                key=lambda x: (
                    x.get('soal_info') is None, 
                    x['soal_info'].get('level', float('inf')) if x.get('soal_info') else float('inf'),
                    x['soal_info'].get('id', float('inf')) if x.get('soal_info') else float('inf'),
                    x['name']
                )
            )
            for f_info in sorted_files:
                # Tampilkan nomor soal dan level dari Excel
                soal_info = f_info.get('soal_info')
                if soal_info:
                    no_soal_txt = f"(Soal no. {soal_info['id']}, level {soal_info['level']})"
                else:
                    no_soal_txt = "<span style='color: orange;'>(Audio tidak terkait di Excel)</span>"
                
                item_cols = st.columns([0.9, 0.1])
                with item_cols[0]:
                    st.markdown(f"<small>{f_info['name']} {no_soal_txt}</small>", unsafe_allow_html=True)
                with item_cols[1]:
                    if st.button("✖", key=f"del_audio_item_{f_info['id']}", help="Hapus file audio ini"):
                        delete_audio_file(f_info['id'])
                        st.rerun()
                st.divider()

    # Tombol untuk menghapus semua data
    if st.button(
        "🗑️ Hapus Semua Data",
        use_container_width=True,
        key="clear_all_main_btn",
        disabled=not (st.session_state.uploaded_files_info or st.session_state.uploaded_question_document_file)
    ):
        clear_all_data()
        st.rerun()

with col_hasil_validasi:
    st.markdown("###### Hasil Validasi Audio")
    with st.container(border=True, height=500):
        if not st.session_state.classification_results and st.session_state.uploaded_files_info:
            st.info("Klik tombol 'Validasi Audio' di bawah untuk memulai proses.")
        elif not st.session_state.classification_results:
            st.info("Belum ada hasil validasi.")
        else:
            audio_map = {f['id']: f for f in st.session_state.uploaded_files_info}
            sorted_results = sorted(
                st.session_state.classification_results,
                key=lambda x: (
                    audio_map.get(x['id'], {}).get('soal_info') is None,
                    audio_map.get(x['id'], {}).get('soal_info', {}).get('level', float('inf')) if audio_map.get(x['id'], {}).get('soal_info') else float('inf'),
                    audio_map.get(x['id'], {}).get('soal_info', {}).get('id', float('inf')) if audio_map.get(x['id'], {}).get('soal_info') else float('inf'),
                    x['filename']
                )
            )
            for res in sorted_results:
                f_info_display = audio_map.get(res['id'])
                # Tampilkan nomor soal dan level dari Excel
                no_soal_res_txt = ""
                if f_info_display:
                    soal_info = f_info_display.get('soal_info')
                    if soal_info:
                        no_soal_res_txt = f"(Soal no. {soal_info['id']}, level {soal_info['level']})"

                st.markdown(f"<b>{res['filename']}</b> <small>{no_soal_res_txt}</small>", unsafe_allow_html=True)
                
                if 'error' in res and res['error']:
                    st.error(f"Error: {res['error']}", icon="❗")
                elif 'probabilities' in res and res['probabilities']:
                    top_cls = max(res['probabilities'], key=res['probabilities'].get)
                    st.write(f"Prediksi: **{top_cls}** (Probabilitas: {res['probabilities'][top_cls]:.2%})")
                    if f_info_display:
                        st.audio(f_info_display['bytes'], format="audio/wav")
                else:
                    st.caption("Menunggu validasi atau tidak ada segmen yang valid.")
                st.divider()

    # Tombol untuk memulai validasi audio
    if st.button(
        "▶️ Validasi Audio",
        use_container_width=True,
        key="validate_main_btn",
        disabled=not st.session_state.uploaded_files_info or not model or st.session_state.get('show_validation_modal')
    ):
        ids_validated = {r['id'] for r in st.session_state.classification_results if 'error' not in r and r.get('probabilities')}
        to_validate = [f for f in st.session_state.uploaded_files_info if f['id'] not in ids_validated]
        if to_validate:
            st.session_state.files_being_validated = to_validate
            st.session_state.show_validation_modal = True
            st.rerun()
        elif st.session_state.uploaded_files_info:
            st.toast("Semua file audio sudah berhasil divalidasi.", icon="✅")
        else:
            st.warning("Tidak ada file audio untuk divalidasi.")


# ==============================================================================
# DIALOG POP-UP
# ==============================================================================
if st.session_state.get('show_export_dialog', False):
    @st.dialog("Buat File JSON Soal", width="large")
    def export_dialog_final_complete():
        st.markdown("### Pratinjau & Edit Soal Sebelum Ekspor")
        st.caption("Untuk soal audio, Anda dapat mengganti kunci jawaban final di bawah ini.")
        st.markdown("---")

        doc_qs = st.session_state.parsed_document_questions
        audio_res_map = {res['filename']: res for res in st.session_state.classification_results if res.get('probabilities')}

        if not doc_qs:
            st.warning("Tidak ada data soal dari Excel untuk ditampilkan.")
            if st.button("Tutup"): st.session_state.show_export_dialog = False; st.rerun()
            return

        # --- BAGIAN 1: TAMPILKAN UI DETAIL & EDIT ---
        with st.container(height=600, border=True):
            for q_data in doc_qs:
                level_num = q_data.get('level_number')
                q_id = q_data.get('id_in_level')
                q_text = q_data.get('teks_soal', "N/A")
                q_opts = q_data.get('opsi_jawaban', {})
                key_excel = str(q_data.get('kunci_jawaban_dokumen', "")).lower()
                audio_file_excel = q_data.get('nama_file_audio_excel')
                
                # UID sekarang lebih sederhana karena tidak ada provinsi
                q_uid_tuple = (level_num, q_id)

                with st.container(border=True):
                    st.subheader(f"Level {level_num} - Soal ID {q_id}")
                    st.markdown(f"**Pertanyaan:** {q_text}")
                    st.markdown("**Opsi Jawaban:**")
                    opt_cols = st.columns(2)
                    for i, (k, v) in enumerate(q_opts.items()):
                        opt_cols[i % 2].markdown(f"&nbsp;&nbsp;**{k.upper()}:** {v}")
                    
                    st.markdown(f"**Kunci Jawaban dari Excel:** `{key_excel.upper()}`")

                    if audio_file_excel:
                        st.markdown(f"**File Audio Terkait:** `{audio_file_excel}`")
                        pred_display = "_Audio belum divalidasi atau tidak ditemukan_"
                        if audio_file_excel in audio_res_map:
                            probs = audio_res_map[audio_file_excel]['probabilities']
                            if probs:
                                top_cls = max(probs, key=probs.get)
                                pred_display = f"Prediksi Model: **{top_cls}** (Prob: {probs[top_cls]:.2%})"
                        st.info(pred_display, icon="�")
                        
                        valid_opts_keys = [k.lower() for k, v in q_opts.items() if v and v.strip()]
                        key_to_show = st.session_state.edited_audio_answer_keys.get(q_uid_tuple, key_excel)
                        
                        try:
                            default_index = valid_opts_keys.index(key_to_show) if key_to_show in valid_opts_keys else 0
                        except ValueError:
                            default_index = 0

                        if valid_opts_keys:
                            edited_key = st.radio(
                                "**Pilih Kunci Jawaban Final:**", valid_opts_keys, 
                                index=default_index, key=f"edit_L{level_num}_Q{q_id}", 
                                horizontal=True, format_func=lambda x: x.upper()
                            )
                            st.session_state.edited_audio_answer_keys[q_uid_tuple] = edited_key

        st.markdown("---")

        # --- BAGIAN 2: PROSES DATA UNTUK OUTPUT JSON NESTED ---
        grouped_levels = defaultdict(list)
        for q_data in doc_qs:
            level_num = q_data.get('level_number')
            if level_num is not None:
                grouped_levels[level_num].append(q_data)

        # Buat satu objek Provinsi (Bali) secara hardcode
        bali_province_object = {
            "nomor_province": 1,
            "nama_province": "Bali",
            "levels_in_province": []
        }

        for level_num, questions_in_level in sorted(grouped_levels.items()):
            level_object = {"level": level_num, "questions_in_level": []}
            for q_data in questions_in_level:
                q_uid_tuple = (q_data.get('level_number'), q_data.get('id_in_level'))
                final_key = st.session_state.edited_audio_answer_keys.get(q_uid_tuple, q_data.get('kunci_jawaban_dokumen'))
                
                img_file = q_data.get('nama_file_gambar')
                audio_file_excel = q_data.get('nama_file_audio_excel')
                
                question_object = {
                    "id": q_data.get('id_in_level'),
                    "path_audio": f"/aset/{audio_file_excel}" if audio_file_excel else None,
                    "path_gambar": f"/aset/{img_file}" if img_file else None,
                    "teks_pertanyaan": q_data.get('teks_soal'),
                    "opsi_jawaban": q_data.get('opsi_jawaban'),
                    "kunci_jawaban_dokumen": final_key
                }
                level_object["questions_in_level"].append(question_object)
            if level_object["questions_in_level"]:
                bali_province_object["levels_in_province"].append(level_object)
        
        final_json_output = [bali_province_object]

        # --- BAGIAN 3: TOMBOL UNDUH & TUTUP ---
        pretty = st.checkbox("Format JSON agar mudah dibaca (Pretty Print)", True)
        json_out = json.dumps(final_json_output, indent=4 if pretty else None, ensure_ascii=False)
        
        dl_col, cc_col = st.columns(2)
        def close_action(): st.session_state.show_export_dialog = False
        
        dl_col.download_button(
            "💾 Unduh susunan_soal.json", json_out, "susunan_soal.json", 
            "application/json", disabled=not final_json_output[0]["levels_in_province"], 
            use_container_width=True, on_click=close_action
        )
        if cc_col.button("Tutup", use_container_width=True):
            close_action()
            st.rerun()

    export_dialog_final_complete()