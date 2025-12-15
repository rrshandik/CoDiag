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

# Ekstrak daftar gejala LANGSUNG dari rule_list.json (SUMBER KEBENARAN)
@st.cache_data
def get_gejala_from_rules():
    """Ekstrak dan urutkan gejala dari rule_list"""
    if not rules_loaded:
        return []
    
    # Kumpulkan semua gejala unik dari rule_list
    gejala_set = set()
    for item in rule_list:
        gejala_set.add(item['gejala'])
    
    # Urutkan alfabetis
    gejala_list = sorted(list(gejala_set))
    
    return gejala_list

gejala_list = get_gejala_from_rules()

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
        
        # penyakit_priors = {
        #     'Lanas': 0.15,
        #     'Phytium_sp': 0.05,
        #     'Ulat_tanah': 0.06,
        #     'Jangkrik': 0.05,
        #     'Kutu_kebul': 0.08,
        #     'Tobacco_mozaic_virus': 0.07,
        #     'Phytophthora_daun': 0.08,
        #     'Begomovirus': 0.06,
        #     'Cucumber_virus': 0.05,
        #     'Virus_kerupuk': 0.05,
        #     'Thrips_parvispinus': 0.12,
        #     'Ulat_grayak': 0.18
        # }

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
                prob_given_no_disease = 0.1  # 10% leak probability (sesuai pakar)
                
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
                        prob_not_symptom = 0.9  # 10% leak probability (sesuai pakar)
                    
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
                <li><strong>Pilih gejala-gejala</strong> yang terlihat pada tanaman Anda di bawah ini</li>
                <li><strong>Klik tombol</strong> "🔍 Mulai Diagnosis"</li>
                <li><strong>Lihat hasil diagnosis</strong> dan tingkat kemungkinannya</li>
            </ol>
            <p><strong>💡 Tips:</strong> Semakin banyak gejala yang dipilih, semakin akurat hasil diagnosisnya</p>
        </div>
        """, unsafe_allow_html=True)

    # Form diagnosis
    with st.form(key=f"diagnosis_form_{st.session_state.form_key}"):
        st.subheader("🔍 Pilih Gejala yang Terlihat")
        
        if gejala_list:
            st.info(f"📝 Total {len(gejala_list)} gejala tersedia dalam sistem")
            
            # Layout 3 kolom untuk checkbox
            cols = st.columns(3)
            evidence_dict = {}
            
            for idx, gejala in enumerate(gejala_list):
                col_idx = idx % 3
                with cols[col_idx]:
                    display_name = format_name(gejala)
                    evidence_dict[gejala] = st.checkbox(
                        display_name, 
                        key=f"symptom_{gejala}_{st.session_state.form_key}"
                    )
            
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
                # Kolom kosong untuk simetri
                pass
                
        else:
            st.error("❌ Tidak dapat memuat daftar gejala!")
            submit_button = False
            reset_button = False

    # Handle reset button
    if reset_button:
        # Increment form_key untuk memaksa form baru dengan state bersih
        st.session_state.form_key += 1
        st.success("✅ Form telah direset!")
        st.rerun()

    # Proses hasil diagnosis
    if submit_button:
        selected_symptoms = [k for k, v in evidence_dict.items() if v]
        
        if not selected_symptoms:
            st.warning("⚠️ Silakan pilih minimal satu gejala untuk melakukan diagnosis!")
        else:
            # Validasi evidence - pastikan gejala ada di model
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