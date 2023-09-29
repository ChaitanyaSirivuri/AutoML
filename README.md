# AutoML

AutoML is a Streamlit application that automates the process of finding the most optimal machine learning model for your dataset. It's a powerful tool for data scientists and machine learning practitioners to streamline the model selection process.

## LIVE LINK

You can access the live application here: [AutoML Live](https://automateml.streamlit.app)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Requirements](#Requirements)

### Installation

To run AutoML locally, follow these installation steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/automl.git
   ```

2. Navigate to the project directory:

   ```bash
   cd automl
   ```

3. Install the required packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Run the application using Streamlit:

   ```bash
   streamlit run app.py
   ```

2. Once the application is running, you can access it in your web browser.

**Choose from the following options in the sidebar:**

- **Upload:** Upload your dataset.
- **Exploratory data analysis:** Perform exploratory data analysis on your dataset.
- **Model Training:** Train machine learning models on your dataset.
- **Download Model:** Download the trained model.

### Features

AutoML offers the following features:

- **Upload:** Upload your dataset to the application.
- **Exploratory data analysis:** Perform exploratory data analysis on your dataset.
- **Model Training:** Train machine learning models on your dataset.
- **Download Model:** Download the trained model.

### Contributing

Contributions to AutoML are welcome! To contribute, follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some feature'`).
5. Push to the branch (`git push origin feature/your-feature`).
6. Create a new Pull Request.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Requirements

Make sure you have the following requirements installed:

- Python 3.6 or higher
- Streamlit (`pip install streamlit`)
- Pandas-Profiling (`pip install pandas-profiling`)
- Streamlit-Pandas-Profiling (`pip install streamlit-pandas-profiling`)
- PyCaret (`pip install pycaret`)

_Note:_ If you don't have the `dataset.csv` file in the root directory, you will need to upload your dataset using the "Upload" option in the sidebar when running the application locally.

Now you can run AutoML locally and explore its capabilities. Enjoy using AutoML for your machine learning projects! 🚀
