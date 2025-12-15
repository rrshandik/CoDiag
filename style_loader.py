import streamlit as st
import os


def load_css(css_file_path):
    """
    Load CSS dari file eksternal dan inject ke Streamlit
    
    Args:
        css_file_path (str): Path ke file CSS
    """
    try:
        with open(css_file_path, 'r', encoding='utf-8') as file:
            css_content = file.read()
        
        # Inject CSS ke Streamlit
        st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
        
    except FileNotFoundError:
        st.warning(f"⚠️ File CSS tidak ditemukan: {css_file_path}")
    except Exception as e:
        st.error(f"❌ Error loading CSS: {e}")


def load_css_from_url(css_url):
    """
    Load CSS dari URL eksternal
    
    Args:
        css_url (str): URL ke file CSS
    """
    st.markdown(
        f'<link rel="stylesheet" href="{css_url}">',
        unsafe_allow_html=True
    )


def apply_theme(theme_name="default"):
    """
    Apply tema CSS yang sudah didefinisikan
    
    Args:
        theme_name (str): Nama tema ('default', 'dark', 'light', 'nature', 'professional')
    """
    theme_classes = {
        'default': '',
        'dark': 'dark-theme',
        'light': 'light-theme', 
        'nature': 'nature-theme',
        'professional': 'professional-theme'
    }
    
    theme_class = theme_classes.get(theme_name, '')
    
    if theme_class:
        st.markdown(
            f'<div class="{theme_class}">',
            unsafe_allow_html=True
        )


def create_custom_css(background_color=None, background_gradient=None, font_family="Inter"):
    """
    Buat CSS kustom secara programatik
    
    Args:
        background_color (str): Warna background solid (hex)
        background_gradient (str): CSS gradient string
        font_family (str): Nama font family
    """
    css_template = f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family={font_family}:wght@300;400;500;600;700&display=swap');
        
        .stApp {{
            {f'background-color: {background_color};' if background_color else ''}
            {f'background: {background_gradient};' if background_gradient else ''}
            font-family: '{font_family}', sans-serif;
        }}
    </style>
    """
    
    st.markdown(css_template, unsafe_allow_html=True)


def load_css_with_fallback(primary_css, fallback_css=None):
    """
    Load CSS dengan fallback jika file utama tidak ada
    
    Args:
        primary_css (str): Path ke file CSS utama
        fallback_css (str): Path ke file CSS cadangan
    """
    if os.path.exists(primary_css):
        load_css(primary_css)
    elif fallback_css and os.path.exists(fallback_css):
        load_css(fallback_css)
        st.info(f"ℹ️ Menggunakan style fallback: {fallback_css}")
    else:
        st.warning("⚠️ Tidak ada file CSS yang dapat dimuat, menggunakan default styling")


# Preset themes untuk kemudahan penggunaan
PRESET_THEMES = {
    'gradient_blue': {
        'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'name': 'Blue Gradient'
    },
    'gradient_nature': {
        'gradient': 'linear-gradient(135deg, #74b9ff 0%, #0984e3 50%, #00cec9 100%)',
        'name': 'Nature Gradient'
    },
    'gradient_warm': {
        'gradient': 'linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%)',
        'name': 'Warm Gradient'
    },
    'solid_light': {
        'color': '#f0f8f0',
        'name': 'Light Green'
    },
    'solid_white': {
        'color': '#ffffff',
        'name': 'White'
    },
    'gradient_dark': {
        'gradient': 'linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%)',
        'name': 'Dark Gradient'
    }
}


def apply_preset_theme(theme_key):
    """
    Apply tema preset yang sudah didefinisikan
    
    Args:
        theme_key (str): Key tema dari PRESET_THEMES
    """
    if theme_key in PRESET_THEMES:
        theme = PRESET_THEMES[theme_key]
        
        if 'gradient' in theme:
            create_custom_css(background_gradient=theme['gradient'])
        elif 'color' in theme:
            create_custom_css(background_color=theme['color'])
        
        return theme['name']
    else:
        st.error(f"❌ Tema tidak ditemukan: {theme_key}")
        return None


def get_available_themes():
    """
    Dapatkan daftar tema yang tersedia
    
    Returns:
        dict: Dictionary tema yang tersedia
    """
    return PRESET_THEMES


# Example usage functions
def demo_theme_selector():
    """
    Demo selector untuk memilih tema
    """
    st.sidebar.subheader("🎨 Theme Selector")
    
    available_themes = get_available_themes()
    theme_options = {v['name']: k for k, v in available_themes.items()}
    
    selected_theme_name = st.sidebar.selectbox(
        "Pilih Tema:",
        options=list(theme_options.keys())
    )
    
    if selected_theme_name:
        theme_key = theme_options[selected_theme_name]
        apply_preset_theme(theme_key)
        st.sidebar.success(f"✅ Tema diterapkan: {selected_theme_name}")


if __name__ == "__main__":
    # Test the functions
    print("Available themes:")
    for key, theme in PRESET_THEMES.items():
        print(f"  {key}: {theme['name']}")