import streamlit as st
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import pandas as pd
import itertools
from collections import defaultdict
import json

# Import style loader
from style_loader import load_css

# Konfigurasi halaman
st.set_page_config(
    page_title="Diagnosis Hama & Penyakit Tembakau",
    page_icon="🌱",
    layout="wide",
)

# Load CSS dari file eksternal
load_css('style/style.css')

# Initialize session state untuk reset functionality
if 'form_key' not in st.session_state:
    st.session_state.form_key = 0

# DEFINISI GRUP GEJALA
SYMPTOM_GROUPS = {
    "Gejala Akar": {
        "icon": "🌿",
        "symptoms": [
            {"display": "Akar Busuk", "value": "Akar_busuk"},
            {"display": "Perlukaan Akar", "value": "Perlukaan_akar"}
        ]
    },
    "Gejala Batang": {
        "icon": "🎋",
        "symptoms": [
            {"display": "Batang Berlubang", "value": "Batang_berlubang"},
            {"display": "Batang Hitam dan Mengkerut", "value": "Batang_hitam_dan_mengkerut"},
            {"display": "Batang Layu", "value": "Batang_layu"},
            {"display": "Batang Terpotong", "value": "Batang_terpotong"}
        ]
    },
    "Gejala Daun - Kondisi Umum": {
        "icon": "🍃",
        "symptoms": [
            {"display": "Daun Layu", "value": "Daun_layu"},
            {"display": "Daun Layu Pada Daerah Terinfeksi", "value": "Daun_layu_pada_daerah_terinfeksi"},
            {"display": "Daun Layu Total", "value": "Daun_layu_total"},
            {"display": "Daun Membusuk", "value": "Daun_membusuk"},
            {"display": "Daun Menyempit", "value": "Daun_menyempit"},
            {"display": "Daun Transparan", "value": "Daun_transparan"}
        ]
    },
    "Gejala Daun - Mengkerut": {
        "icon": "🥬",
        "symptoms": [
            {"display": "Daun Mengkerut (Umum)", "value": "Daun_mengkerut"},
            {"display": "Daun Mengkerut Berwarna Kuning Keemasan", "value": "Daun_mengkerut_berwarna_kuning_keemasan"},
            {"display": "Daun Mengkerut dengan Warna Tidak Rata", "value": "Daun_mengkerut_dengan_warna_tidak_rata"},
            {"display": "Daun Mengkerut Seperti Kerupuk", "value": "Daun_mengkerut_seperti_kerupuk"}
        ]
    },
    "Gejala Daun - Bercak & Warna": {
        "icon": "🎨",
        "symptoms": [
            {"display": "Bawah Daun Bercak Hitam", "value": "Bawah_daun_bercak_hitam"},
            {"display": "Daun Bercak Hijau Gelap Tidak Merata", "value": "Daun_bercak_hijau_gelap_tidak_merata"},
            {"display": "Daun Berkerut Mozaik", "value": "Daun_berkerut_mozaik"},
            {"display": "Warna Daun Memudar", "value": "Warna_daun_memudar"},
            {"display": "Terdapat Bercak Putih Kecoklatan di Sekitar Tulang Daun", "value": "Terdapat_bercak_putih_kecoklatan_di_sekitar_tulang_daun"}
        ]
    },
    "Gejala Daun - Kerusakan Fisik": {
        "icon": "🦗",
        "symptoms": [
            {"display": "Daun Berlubang", "value": "Daun_berlubang"},
            {"display": "Terdapat Banyak Lubang Kecil", "value": "Terdapat_banyak_lubang_kecil"},
            {"display": "Terdapat Bekas Gigitan pada Jaringan Daun", "value": "Terdapat_bekas_gigitan_pada_jaringan_daun"}
        ]
    },
    "Gejala Tanaman Secara Keseluruhan": {
        "icon": "🌱",
        "symptoms": [
            {"display": "Tanaman Kerdil", "value": "Tanaman_kerdil"},
            {"display": "Terdapat Pola Jaring Laba-laba Berwarna Kuning Kehitaman", "value": "Terdapat_pola_jaring_laba-laba_berwarna_kuning_kehitaman"}
        ]
    }
}

# Load data dari file JSON
@st.cache_data
def load_json_data():
    try:
        with open('rule_list.json', 'r', encoding='utf-8') as f:
            rule_list = json.load(f)
        
        with open('individuals_list.json', 'r', encoding='utf-8') as f:
            individuals_list = json.load(f)
        
        return rule_list, individuals_list, True
    except FileNotFoundError as e:
        st.error(f"File JSON tidak ditemukan: {e}")
        return [], [], False
    except json.JSONDecodeError as e:
        st.error(f"Error parsing JSON: {e}")
        return [], [], False

# Load data
rule_list, individuals_list, rules_loaded = load_json_data()

# Daftar semua hama dan penyakit
hama_penyakit_list = [
    "Lanas", "Phytium_sp", "Ulat_tanah", "Jangkrik",
    "Kutu_kebul", "Tobacco_mozaic_virus", "Phytophthora_daun",
    "Begomovirus", "Cucumber_virus", "Virus_kerupuk", 
    "Thrips_parvispinus", "Ulat_grayak" 
]

# Fungsi untuk membangun model Bayesian Network
@st.cache_resource
def build_bayesian_model():
    if not rules_loaded:
        return None, None
    
    try:
        # Buat struktur jaringan
        edges = []
        for item in rule_list:
            edges.append((item['nama'], item['gejala']))

        model = DiscreteBayesianNetwork(edges)

        # Set prior probability untuk setiap penyakit/hama
        penyakit_priors = {
            'Lanas': 0.5,
            'Phytium_sp': 0.5,
            'Ulat_tanah': 0.5,
            'Jangkrik': 0.5,
            'Kutu_kebul': 0.5,
            'Tobacco_mozaic_virus': 0.5,
            'Phytophthora_daun': 0.5,
            'Begomovirus': 0.5,
            'Cucumber_virus': 0.5,
            'Virus_kerupuk': 0.5,
            'Thrips_parvispinus': 0.5,
            'Ulat_grayak': 0.5
        }

        # Masukkan CPT untuk node penyakit/hama (prior)
        penyakit = set([item['nama'] for item in rule_list])
        for p in penyakit:
            prior_prob = penyakit_priors.get(p, 0.1)
            cpd_p = TabularCPD(variable=p, variable_card=2, 
                              values=[[1-prior_prob], [prior_prob]])
            model.add_cpds(cpd_p)

        # Group gejala berdasarkan parent nodes
        gejala_to_parents = defaultdict(list)
        gejala_to_scores = defaultdict(dict)

        for item in rule_list:
            g = item['gejala']
            p = item['nama']
            gejala_to_parents[g].append(p)
            gejala_to_scores[g][p] = float(item['skor'])

        # Masukkan CPT untuk gejala
        for g, parents in gejala_to_parents.items():
            if len(parents) == 1:
                p = parents[0]
                prob_given_disease = gejala_to_scores[g][p]
                prob_given_no_disease = 0.1
                
                cpd_g = TabularCPD(variable=g, variable_card=2, 
                                  values=[[1 - prob_given_no_disease, 1 - prob_given_disease],
                                          [prob_given_no_disease, prob_given_disease]],
                                  evidence=[p], evidence_card=[2])
                model.add_cpds(cpd_g)
            else:
                # Multiple parents dengan noisy-OR
                n_parents = len(parents)
                values_0 = []
                values_1 = []
                
                for parent_states in itertools.product([0, 1], repeat=n_parents):
                    prob_not_symptom = 1.0
                    
                    for idx, state in enumerate(parent_states):
                        p = parents[idx]
                        if state == 1:
                            prob_not_symptom *= (1 - gejala_to_scores[g][p])
                    
                    if sum(parent_states) == 0:
                        prob_not_symptom = 0.9
                    
                    prob_symptom = 1 - prob_not_symptom
                    values_0.append(1 - prob_symptom)
                    values_1.append(prob_symptom)
                
                cpd_g = TabularCPD(variable=g, variable_card=2,
                                   values=[values_0, values_1],
                                   evidence=parents, evidence_card=[2]*n_parents)
                model.add_cpds(cpd_g)

        # Validasi model
        model.check_model()
        return model, VariableElimination(model), penyakit_priors
        
    except Exception as e:
        st.error(f"Error membangun model: {e}")
        return None, None, None

# Helper functions
def format_name(name):
    """Format nama gejala"""
    return name.replace('_', ' ').title()

def display_results(selected_symptoms, posterior_probs):
    """Tampilkan hasil diagnosis"""
    st.markdown('<div class="result-section">', unsafe_allow_html=True)
    st.subheader("📊 Hasil Diagnosis")
    
    # Gejala yang dipilih
    st.markdown("**🔍 Gejala yang dipilih:**")
    symptoms_display = ", ".join([format_name(s) for s in selected_symptoms])
    st.write(symptoms_display)
    
    st.markdown("---")
    
    # Urutkan hasil
    sorted_results = sorted(posterior_probs.items(), key=lambda x: x[1], reverse=True)
    
    if any(prob > 0 for _, prob in sorted_results):
        st.markdown("**🎯 Kemungkinan Hama/Penyakit:**")
        
        # Tampilkan top 5 hasil
        for i, (hp, prob) in enumerate(sorted_results[:5]):
            if prob > 0:
                nama_display = format_name(hp)
                percentage = prob * 100
                
                # Progress bar
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{i+1}. {nama_display}**")
                    st.progress(prob)
                with col2:
                    st.write(f"{percentage:.1f}%")
                
                st.markdown("---")
        
        # Tabel detail lengkap
        st.subheader("📋 Detail Lengkap")
        df_results = []
        for hp, prob in sorted_results:
            if prob > 0:
                nama = format_name(hp)
                persen = f"{prob*100:.1f}%"
                df_results.append([nama, persen])
        
        if df_results:
            df = pd.DataFrame(df_results, columns=["Hama/Penyakit", "Probabilitas"])
            st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("ℹ️ Tidak ada diagnosis yang dapat ditentukan berdasarkan gejala yang dipilih.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main application
def main():
    """Function utama aplikasi"""
    
    # Load model
    model_data = build_bayesian_model()
    
    if model_data is None or model_data[0] is None:
        st.error("❌ Tidak dapat memuat model diagnosis. Pastikan file rule_list.json dan individuals_list.json tersedia.")
        return
    
    model, infer, penyakit_priors = model_data
    
    # Header
    st.markdown('<h1 class="main-title">Sistem Diagnosis Hama & Penyakit Tembakau</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Pilih gejala yang Anda amati pada tembakau untuk mendapatkan diagnosis yang akurat</p>', unsafe_allow_html=True)

    # Info box
    with st.container():
        st.markdown("""
        <div class="info-container">
            <h4>📋 Cara Menggunakan:</h4>
            <ol>
                <li><strong>Pilih kategori gejala</strong> yang relevan dengan kondisi tanaman Anda</li>
                <li><strong>Centang gejala-gejala spesifik</strong> yang terlihat pada tanaman</li>
                <li><strong>Klik tombol</strong> "🔍 Mulai Diagnosis"</li>
                <li><strong>Lihat hasil diagnosis</strong> dan tingkat kemungkinannya</li>
            </ol>
            <p><strong>💡 Tips:</strong> Gejala telah dikelompokkan berdasarkan bagian tanaman untuk memudahkan pemilihan</p>
        </div>
        """, unsafe_allow_html=True)

    # Form diagnosis
    with st.form(key=f"diagnosis_form_{st.session_state.form_key}"):
        st.subheader("🔍 Pilih Gejala yang Terlihat")
        
        evidence_dict = {}
        
        # Tampilkan gejala berdasarkan grup
        for group_name, group_data in SYMPTOM_GROUPS.items():
            with st.expander(f"{group_data['icon']} **{group_name}** ({len(group_data['symptoms'])} gejala)", expanded=True):
                # Layout 2 kolom untuk setiap grup
                cols = st.columns(2)
                
                for idx, symptom in enumerate(group_data['symptoms']):
                    col_idx = idx % 2
                    with cols[col_idx]:
                        evidence_dict[symptom['value']] = st.checkbox(
                            symptom['display'],
                            key=f"symptom_{symptom['value']}_{st.session_state.form_key}"
                        )
        
        # Info jumlah gejala total
        total_symptoms = sum(len(group['symptoms']) for group in SYMPTOM_GROUPS.values())
        st.info(f"📝 Total {total_symptoms} gejala tersedia dalam {len(SYMPTOM_GROUPS)} kategori")
        
        # Tombol submit dan reset dalam form
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            reset_button = st.form_submit_button(
                label="🔄 Reset",
                type="secondary",
                use_container_width=True
            )
        
        with col2:
            submit_button = st.form_submit_button(
                label="🔍 Mulai Diagnosis",
                type="primary",
                use_container_width=True
            )
        
        with col3:
            pass

    # Handle reset button
    if reset_button:
        st.session_state.form_key += 1
        st.success("✅ Form telah direset!")
        st.rerun()

    # Proses hasil diagnosis
    if submit_button:
        selected_symptoms = [k for k, v in evidence_dict.items() if v]
        
        if not selected_symptoms:
            st.warning("⚠️ Silakan pilih minimal satu gejala untuk melakukan diagnosis!")
        else:
            # Validasi evidence
            all_nodes = set(model.nodes())
            valid_symptoms = [s for s in selected_symptoms if s in all_nodes]
            invalid_symptoms = [s for s in selected_symptoms if s not in all_nodes]
            
            if invalid_symptoms:
                st.warning(f"⚠️ Gejala berikut tidak ditemukan di model: {', '.join([format_name(s) for s in invalid_symptoms])}")
            
            if not valid_symptoms:
                st.error("❌ Tidak ada gejala valid yang ditemukan di model!")
            else:
                with st.spinner("🔄 Sedang menganalisis gejala..."):
                    # Hitung probabilitas
                    posterior_probs = {}
                    evidence = {symptom: 1 for symptom in valid_symptoms}
                    
                    # Hitung posterior untuk setiap hama/penyakit
                    for hp in hama_penyakit_list:
                        try:
                            result = infer.query(variables=[hp], evidence=evidence)
                            posterior_probs[hp] = result.values[1]
                        except Exception as e:
                            posterior_probs[hp] = 0
                    
                    # Tampilkan hasil
                    display_results(valid_symptoms, posterior_probs)

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🌱 <strong>Sistem Diagnosis Hama & Penyakit Tembakau</strong></p>
        <p>Built with Knowledge Graph and Bayesian Network</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()