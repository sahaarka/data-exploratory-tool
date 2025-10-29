<<<<<<< HEAD
# data-exploratory-tool
A readymade data analysis tool for data exploration/discovery use case
=======
# 🧭 Data Discovery Tool

The Data Discovery Tool is an interactive Python application for exploring and analyzing datasets with minimal effort.
It allows users to upload CSV files, automatically detect data types, parse dates, generate visual insights, identify potential join keys, and perform advanced statistical and pattern-based analysis — all through a rich Streamlit web interface.

🛡️ Note: This version of the tool is obfuscated using PyArmor to protect the source code.

## 🚀 Key Features

* Large File Handling: Efficiently processes large CSV files in chunks, supporting files well over 500 MB.
* Smart Data Type Detection: Automatically classifies columns as numeric, date, categorical, or text.
* Robust Date Parsing: Detects and converts common date formats automatically.
* Adaptive Visualizations: Generates context-aware charts — histograms, box plots, bar and pie charts — based on column type.
* Dataset Management: Upload, view, and delete datasets within the app sidebar.
* Join Key Analysis: Detects potential join keys between datasets using configurable match thresholds.
* Statistical Analysis: Computes both descriptive and advanced statistics (mean, skewness, entropy, etc.).
* Pattern Extraction: Derives and generalizes regex-like patterns to understand data structure and consistency.
* Univariate Analysis: Includes detailed per-column insights — metadata, statistics, distribution, and pattern analysis.
* Multivariate Analysis: Enables correlation and pairwise visualization between numeric columns.
* Dataset Preview: Provides random row previews and column type summaries.
* Streamlit-based UI: Interactive, responsive interface with custom CSS for a polished look.
* Scrolling Footer: Displays copyright and attribution.


## ⚙️ Dependencies

The following Python packages are automatically installed when the tool is installed via the wheel:

* streamlit
* pandas
* numpy
* plotly
* seaborn
* matplotlib
* scikit-learn

(Modules like re, collections, math, etc. are part of the Python standard library and do not need to be installed.)

## 🧩 Installation & Execution
### 1️⃣ Download the Distribution
Download the data-discovery-tool.zip file, which contains the prebuilt .whl package.

### 2️⃣ Extract the Package
Extract the contents of the ZIP file to a folder of your choice.

### 3️⃣ Install the Wheel
Open a terminal or command prompt, navigate to the extracted folder, and run:
```bash
pip install data_discovery_tool-0.1.0-py3-none-any.whl
```

    💡 Replace the filename with the exact .whl version included in your package if it differs.

### 4️⃣ Run the Application (Two Options)
#### ✅ Option 1 — Recommended (Run via CLI command)

After installation, launch the app using the built-in console script:
```bash
data-discovery-tool
```
This will start the Streamlit server automatically.
Open your browser and navigate to:

👉 http://localhost:8501

#### 🧰 Option 2 — Run Directly via Streamlit Command

If you want to run the app manually without using the CLI entry point (e.g., during development or debugging), you can directly execute:
```bash
streamlit run app.py
```

    💡 Run this command from the same directory where app.py is located.

You can also customize options, for example:
```bash
streamlit run app.py --server.port=8502 --server.maxUploadSize=1024
```

### 🖥️ OS-Specific Notes
#### 🪟 Windows

Ensure Python and pip are added to your system PATH.

For any visualization library issues (e.g., Matplotlib), consider using Anaconda or installing dependencies manually.

#### 🍎 macOS

Make sure you have Xcode Command Line Tools installed:
```bash
xcode-select --install
```

If you encounter backend issues with Matplotlib, set:
```bash
import matplotlib
matplotlib.use('TkAgg')
```

#### 🐧 Linux

Ensure system dependencies for Matplotlib are available:
```bash
sudo apt-get install python3-tk
```

Using a virtual environment (venv or conda) is strongly recommended to avoid conflicts with system Python.

🗂️ Project Structure (Inside the Extracted Zip)
data_discovery_tool/
│
|├── app.py
|├── cli.py
|├── data_tool.py
|├── __init__.py
├── setup.py

🔧 Troubleshooting & Tips
Issue	Possible Fix
Port 8501 already in use	Run with another port:
streamlit run app.py --server.port=8502
Blank or partially loaded page	Clear browser cache and refresh.
Package not found after install	Ensure you installed the correct .whl file:
pip install dist/data_discovery_tool-0.1.0-py3-none-any.whl
App not opening in browser	Open manually: http://localhost:8501
🧠 Author

Arka Saha
Data Engineering Professional | Accenture
📧 anticlock909@gmail.com
## Features

-   **Large File Handling:** Efficiently processes large CSV files by loading them in chunks.
-   **Intelligent Data Type Detection:** Automatically categorizes columns as numeric, date, categorical, or text.
-   **Date Parsing:** Intelligently detects and parses date columns in various formats.
-   **Smart Visualizations:** Generates visualizations tailored to the detected data types, including histograms, box plots, bar charts, and pie charts.
-   **Dataset Management:** Allows users to view and delete uploaded datasets.
-   **Streamlit-based Interface:** Provides an interactive and user-friendly web interface.
-   **Potential Join Key Analysis:** Finds potential join keys between datasets with match percentage analysis.
-   **Statistical Analysis:** Computes basic and advanced statistics for different data types.
-   **Pattern Extraction:** Generates and generalizes regex-like patterns for string columns.
-   **Large file configuration:** Allows user to change delimiter, header detection, sampling, and chunk sizes.
-   **Univariate Analysis:** Provides detailed analysis of individual columns, including metadata, statistical summaries, distribution visualizations, advanced insights, and pattern analysis.
-   **Multivariate Analysis:** Performs correlation and pairwise relationship analysis between numeric columns.
-   **Dataset Preview:** Allows users to view a preview of their uploaded datasets, including random rows and column type distributions.
-   **Custom CSS Styling:** Provides an enhanced user interface with custom CSS styling for the sidebar, header, and footer.
-   **Scrolling Footer:** Includes a scrolling footer with copyright and powered-by information.
-   **Obfuscated Source Code:** Protected using PyArmor to secure the source code.

## Dependencies

-   streamlit
-   pandas
-   numpy
-   plotly
-   seaborn
-   matplotlib
-   scikit-learn
-   re
-   collections

## Installation and Execution

1.  **Download the Distribution Package:**
    * Download the `data-discovery-tool.zip` file containing the installation package.

2.  **Extract the Package:**
    * Extract the contents of the `data-discovery-tool.zip` file to a directory on your computer.

3.  **Install the Package:**
    * Open a command prompt or terminal.
    * Navigate to the directory where you extracted the `data-discovery-tool.zip` file.
    * Run the following command to install the package:

        ```bash
        pip install data_discovery_tool-0.1.0-py3-none-any.whl  # Replace with the actual filename
        ```

        **Note:** The Streamlit app runs in your local and the address is http://localhost:8501

    **OS-Specific Notes:**

    * **Windows:**
        * Ensure Python and `pip` are added to your system's PATH environment variables.
        * If you encounter issues with libraries like `matplotlib`, consider installing pre-built binaries or using a Python distribution like Anaconda.
    * **macOS:**
        * Ensure you have Xcode Command Line Tools installed (`xcode-select --install`).
        * If you use Homebrew, ensure it's up to date.
        * For potential issues with `matplotlib` backend, you might need to configure it (`matplotlib.use('TkAgg')` or similar).
    * **Linux:**
        * Ensure you have the necessary system dependencies for libraries like `matplotlib` (`sudo apt-get install python3-tk` on Debian/Ubuntu, or equivalent for your distribution).
        * Virtual environments (`venv` or `conda`) are highly recommended to avoid conflicts with system Python packages.

4.  **Run the Application:**
    * After installation, you can run the application using the entry point:

        ```bash
        data-discovery-tool
        ```

## Project Structure (Inside the Extracted Zip)
>>>>>>> 152832f... initial version of the data exploratory tool
