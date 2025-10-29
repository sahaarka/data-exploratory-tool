import streamlit as st
try:
    from .data_tool import DataDiscoveryTool  # When imported as a module
except ImportError:
    from data_tool import DataDiscoveryTool  # When run directly


def run_app():


    # Set maxUploadSize here
    # st.set_option('server.maxUploadSize', 1024) # 1024 MB
    # print("Starting Data Discovery Tool...")
    # Set page configuration
    st.set_page_config(
        page_title="Data Discovery Tool",
        page_icon="📊",
        layout="wide"
    )


    sidebar_style = """
        <style>
            [data-testid="stSidebar"] {
                background: linear-gradient(135deg, #e6e9ff 0%, #d4d7fc 50%, #c4c8fa 100%);
                border-right: 1px solid #b6bbf3;
                box-shadow: 3px 0px 15px rgba(0, 0, 0, 0.12);
                padding-top: 1rem;
            }
           
            [data-testid="stSidebar"]::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0));
                pointer-events: none;
            }
           
            [data-testid="stSidebar"] h2 {
                color: #4e54c8;
                letter-spacing: 0.5px;
                font-weight: 600;
                text-shadow: 1px 1px 1px rgba(255, 255, 255, 0.7);
            }
           
            [data-testid="stSidebar"] .stRadio label,
            [data-testid="stSidebar"] .stCheckbox label,
            [data-testid="stSidebar"] .stSelectbox label {
                font-weight: 500;
                color: #3a3f99;
            }
        </style>
    """
    st.markdown(sidebar_style, unsafe_allow_html=True)
   
    # Initialize and run the tool
    tool = DataDiscoveryTool()
    tool.main()


    # Custom CSS to fix the footer at the bottom
    footer_css = """
    <style>
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background: linear-gradient(to right, #4e54c8, #8f94fb);
        box-shadow: 0 -4px 6px rgba(0, 0, 0, 0.1);
        color: white;
        text-align: center;
        padding: 1px;  /* Reduced padding for a thinner look */
        font-size: 12px;  /* Slightly smaller font */
        font-weight: bold;
        z-index: 100;
        height: 20px;  /* Control the overall height of the footer */
        line-height: 20px; /* Ensure text aligns well inside */
    }
    </style>
    """


    # Scrolling text with Marquee
    scrolling_footer_html = """
    <div class="footer">
        <marquee behavior="scroll" direction="left" scrollamount="5">
            © 2025 Arka Saha. All rights reserved. | Powered by Streamlit 🚀
        </marquee>
    </div>
    """


    # Render the footer
    st.markdown(footer_css, unsafe_allow_html=True)
    st.markdown(scrolling_footer_html, unsafe_allow_html=True)


if __name__ == '__main__':
    run_app()