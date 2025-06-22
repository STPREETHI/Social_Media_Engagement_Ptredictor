import streamlit as st
import pandas as pd  # Global pandas import - DO NOT REMOVE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
from datetime import datetime
import time

def render_multi_tab_ui(processor):
    """Render multi-tab interface"""
    st.title("Social Media Engagement Prediction App")
    
    # Create tabs
    tabs = st.tabs([
        "📤 Data Upload", 
        "🔍 Preprocessing",
        "📊 Visualization",
        "🔮 Prediction",
        "📋 Classification",
        "📍 Clustering",
        "📈 Forecasting",
        "📅 Yearly Sentiment",
        "📊 Hypothesis Tests"
    ])
    
    # Data Upload tab
    with tabs[0]:
        render_data_upload_tab(processor)
    
    # Preprocessing tab
    with tabs[1]:
        render_preprocessing_tab(processor)
    
    # Visualization tab
    with tabs[2]:
        render_visualization_tab(processor)
    
    # Prediction tab
    with tabs[3]:
        render_prediction_tab(processor)
    
    # Classification tab
    with tabs[4]:
        render_classification_tab(processor)
    
    # Clustering tab
    with tabs[5]:
        render_clustering_tab(processor)
    
    # Forecasting tab
    with tabs[6]:
        render_forecasting_tab(processor)
    
    # Yearly Sentiment tab
    with tabs[7]:
        render_yearly_sentiment_tab(processor)
        
    # Hypothesis Testing tab
    with tabs[8]:
        render_hypothesis_testing_tab(processor)

def render_data_upload_tab(processor):
    """Render data upload section"""
    st.header("Data Upload")
    
    st.markdown("""
    This application predicts social media sentiment and engagement using machine learning models. 
    Upload your data to start analyzing trends and making predictions.
    """)
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Load data
            if processor.load_data(uploaded_file):
                st.session_state.file_uploaded = True
                st.success("Data uploaded successfully!")
                
                # Display data preview
                st.subheader("Preview of the uploaded data")
                st.dataframe(processor.df.head())
                
                # Display data info
                data_info = processor.get_data_info()
                if data_info:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Dataset Shape")
                        st.write(f"Rows: {data_info['shape'][0]}")
                        st.write(f"Columns: {data_info['shape'][1]}")
                    
                    with col2:
                        st.subheader("Missing Values")
                        missing_df = pd.DataFrame.from_dict(data_info['missing_values'], 
                                                           orient='index',
                                                           columns=['Count'])
                        missing_df = missing_df[missing_df['Count'] > 0]
                        if not missing_df.empty:
                            st.dataframe(missing_df)
                        else:
                            st.write("No missing values!")
            else:
                st.error("Failed to load data. Please check your file.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        # Only show this message on first load
        if not st.session_state.get('file_uploaded', False):
            st.info("Please upload a CSV file to begin.")
            
            # Option to use demo data
            use_demo = st.checkbox("Use demo data")
            if use_demo:
                st.info("Loading demo data...")
                # Create sample data with proper timestamp format
                from datetime import datetime, timedelta
                
                # Create a demo dataset
                n_samples = 100
                base_date = datetime(2023, 1, 1)
                dates = [base_date + timedelta(days=i) for i in range(n_samples)]
                
                platforms = ['Twitter', 'Instagram', 'Facebook', 'LinkedIn', 'TikTok']
                sentiments = ['Positive', 'Neutral', 'Negative']
                
                # Generate random data
                demo_data = {
                    'Timestamp': [d.strftime('%d-%m-%Y %H:%M') for d in dates],
                    'Platform': np.random.choice(platforms, n_samples),
                    'Likes': np.random.randint(0, 1000, n_samples),
                    'Retweets': np.random.randint(0, 300, n_samples),
                    'Sentiment': np.random.choice(sentiments, n_samples),
                    'Hashtags': [f"#{np.random.choice(['tech', 'news', 'trending', 'viral'])}" 
                                for _ in range(n_samples)]
                }
                
                # Create DataFrame
                demo_df = pd.DataFrame(demo_data)
                
                # Save to a temporary CSV file in memory
                from io import StringIO
                csv_buffer = StringIO()
                demo_df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)
                
                # Load data
                if processor.load_data(csv_buffer):
                    st.session_state.file_uploaded = True
                    st.success("Demo data loaded successfully!")
                    
                    # Display data preview
                    st.subheader("Preview of the demo data")
                    st.dataframe(processor.df.head())
                    
                    # Preprocess the data
                    processor.preprocess()
                    
                    # Display data info
                    data_info = processor.get_data_info()
                    if data_info:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Dataset Shape")
                            st.write(f"Rows: {data_info['shape'][0]}")
                            st.write(f"Columns: {data_info['shape'][1]}")
                        
                        with col2:
                            st.subheader("Data Overview")
                            if 'Platform' in processor.df.columns:
                                st.write(f"Platforms: {len(processor.df['Platform'].unique())}")
                            if 'Timestamp' in processor.df.columns and not processor.df['Timestamp'].isna().all():
                                st.write(f"Date Range: {processor.df['Timestamp'].min().date()} to {processor.df['Timestamp'].max().date()}")
                            else:
                                st.write("Demo data loaded successfully.")
                else:
                    st.error("Failed to load demo data.")


def render_preprocessing_tab(processor):
    """Render preprocessing section"""
    st.header("Data Preprocessing")
    
    if not processor.is_data_loaded:
        st.warning("Please upload data in the Data Upload tab first.")
        return
    
    st.subheader("Original Data")
    st.dataframe(processor.raw_df.head())
    
    # Preprocessing steps
    steps_container = st.container()
    
    with steps_container:
        st.subheader("Preprocessing Steps")
        
        # Missing value handling
        st.write("1. Missing Value Handling")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Before (null values):")
            missing_before = processor.raw_df.isnull().sum()
            missing_before = missing_before[missing_before > 0]
            if not missing_before.empty:
                st.dataframe(missing_before)
            else:
                st.write("No missing values!")
        
        with col2:
            if st.button("Handle Missing Values"):
                # Preprocessing
                processor.preprocess()
                
                st.write("After (null values):")
                missing_after = processor.df.isnull().sum()
                missing_after = missing_after[missing_after > 0]
                if not missing_after.empty:
                    st.dataframe(missing_after)
                else:
                    st.write("No missing values!")
        
        # Feature engineering
        st.write("2. Feature Engineering")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Before (original features):")
            st.dataframe(processor.raw_df.head(3))
        
        with col2:
            if st.button("Apply Feature Engineering"):
                # Ensure preprocessing is done
                processor.preprocess()
                
                st.write("After (new features):")
                # Show only the new features
                new_features = ['Hashtag_Count', 'Engagement_Rate', 'Engagement_Category']
                new_features = [f for f in new_features if f in processor.df.columns]
                if new_features:
                    st.dataframe(processor.df[new_features].head(3))
                else:
                    st.write("No new features created.")
        
        # Encoding
        st.write("3. Categorical Encoding")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Before (categorical features):")
            categorical_cols = processor.raw_df.select_dtypes(include=['object']).columns
            if not categorical_cols.empty:
                st.dataframe(processor.raw_df[categorical_cols].head(3))
            else:
                st.write("No categorical features!")
        
        with col2:
            if st.button("Apply Encoding"):
                # Ensure preprocessing is done
                processor.preprocess()
                
                st.write("After (encoded features):")
                encoded_cols = [col for col in processor.df.columns if col.endswith('_Encoded') 
                               or col.startswith('Platform_')]
                if encoded_cols:
                    st.dataframe(processor.df[encoded_cols].head(3))
                else:
                    st.write("No encoded features created.")
    
    # Final preprocessed data
    st.subheader("Final Preprocessed Data")
    if st.button("Complete All Preprocessing"):
        # Apply all preprocessing steps
        processor.preprocess()
        st.success("Preprocessing completed!")
        st.dataframe(processor.df.head())

def render_visualization_tab(processor):
    """Render data visualization section"""
    st.header("Data Visualization")
    
    if not processor.is_data_loaded:
        st.warning("Please upload data in the Data Upload tab first.")
        return
    
    # Apply preprocessing if not done yet
    if 'Engagement_Category' not in processor.df.columns:
        if st.button("Preprocess Data for Visualization"):
            processor.preprocess()
            st.success("Data preprocessed successfully!")
    
    # Visualization types
    viz_type = st.selectbox(
        "Select visualization",
        ["Correlation Matrix", "Sentiment Distribution", "Platform Distribution", 
         "Engagement Trend", "Engagement Categories"]
    )
    
    if viz_type == "Correlation Matrix":
        st.subheader("Correlation Matrix")
        fig = processor.get_correlation_plot()
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Unable to create correlation matrix.")
    
    elif viz_type == "Sentiment Distribution":
        st.subheader("Sentiment Distribution")
        fig = processor.get_sentiment_distribution_plot()
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Sentiment column not found in the dataset.")
    
    elif viz_type == "Platform Distribution":
        st.subheader("Platform Distribution")
        fig = processor.get_platform_distribution_plot()
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Platform column not found in the dataset.")
    
    elif viz_type == "Engagement Trend":
        st.subheader("Engagement Trend")
        fig = processor.get_engagement_trend_plot()
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Timestamp column not found or cannot be processed.")
    
    elif viz_type == "Engagement Categories":
        st.subheader("Engagement Categories")
        fig = processor.get_engagement_category_plot()
        if fig:
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Engagement Category not found. Apply preprocessing first.")

def render_prediction_tab(processor):
    """Render prediction section"""
    st.header("Engagement Prediction")
    
    if not processor.is_data_loaded:
        st.warning("Please upload data in the Data Upload tab first.")
        return
    
    # Check if model is trained
    if not hasattr(processor, 'trained_models') or not processor.trained_models:
        st.warning("Models are not trained yet. Please train models in the Classification tab first.")
        if st.button("Train Models Now"):
            # Ensure data is preprocessed and prepared
            processor.preprocess()
            processor.prepare_model_data()
            
            # Train models
            with st.spinner("Training models..."):
                if processor.train_models():
                    st.success("Models trained successfully!")
                else:
                    st.error("Failed to train models.")
        return
    
    st.subheader("Enter values to predict engagement:")
    
    # Create input form
    col1, col2 = st.columns(2)
    
    with col1:
        likes = st.number_input("Likes", min_value=0, value=100)
        
        # Determine available sentiments
        sentiments = ["Positive", "Neutral", "Negative"]
        if 'Sentiment' in processor.df.columns:
            sentiments = processor.df['Sentiment'].unique().tolist()
        
        sentiment = st.selectbox("Sentiment", options=sentiments)
    
    with col2:
        retweets = st.number_input("Retweets", min_value=0, value=50)
        hashtag_count = st.number_input("Hashtag Count", min_value=0, value=2)
    
    # Create input data
    input_data = {
        'Likes': likes,
        'Retweets': retweets,
        'Hashtag_Count': hashtag_count,
        'Sentiment': sentiment
    }
    
    if st.button("Predict Engagement"):
        with st.spinner("Making prediction..."):
            result = processor.predict_engagement(input_data)
            
            if result:
                st.success("Prediction Complete!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Engagement", result['category'])
                    
                    # Get class names from the result or use default order
                    class_names = result.get('classes', ['Low', 'Medium', 'High'])
                    probabilities = result['probabilities']
                    
                    # Create dataframe for probabilities
                    prob_df = pd.DataFrame({
                        'Category': class_names,
                        'Probability': probabilities
                    })
                    
                    # Sort by probability in descending order to show most likely first
                    prob_df = prob_df.sort_values('Probability', ascending=False)
                    
                    # Highlight the predicted category
                    st.markdown(f"**Most likely category: {result['category']} with {prob_df.iloc[0]['Probability']*100:.1f}% probability**")
                    st.dataframe(prob_df)
                
                with col2:
                    # Create bar chart for probabilities - use sorted values for better visibility
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    # Use specific colors for each class, consistent with category names
                    colors = {'Low': '#ff9999', 'Medium': '#66b3ff', 'High': '#99ff99'}
                    bar_colors = [colors[cat] for cat in prob_df['Category']]
                    
                    ax.bar(prob_df['Category'], prob_df['Probability'], color=bar_colors)
                    ax.set_ylim(0, 1)
                    ax.set_title("Prediction Probabilities")
                    ax.set_ylabel("Probability")
                    
                    # Add annotation for the predicted class
                    highest_idx = prob_df['Probability'].argmax()
                    ax.annotate(f"{prob_df.iloc[highest_idx]['Probability']*100:.1f}%", 
                               xy=(highest_idx, prob_df.iloc[highest_idx]['Probability']),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.error("Failed to make prediction.")

def render_classification_tab(processor):
    """Render classification section"""
    st.header("Classification Models")
    
    if not processor.is_data_loaded:
        st.warning("Please upload data in the Data Upload tab first.")
        return
    
    st.subheader("Train Classification Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Prepare Data for Modeling"):
            with st.spinner("Preprocessing and preparing data..."):
                # Preprocess data if needed
                if 'Engagement_Category' not in processor.df.columns:
                    processor.preprocess()
                
                # Prepare for modeling
                if processor.prepare_model_data():
                    st.success("Data prepared for modeling!")
                    
                    # Show feature info
                    st.write(f"Features: {', '.join(processor.feature_names)}")
                    st.write(f"Training set size: {len(processor.X_train)} samples")
                    st.write(f"Test set size: {len(processor.X_test)} samples")
                else:
                    st.error("Failed to prepare data for modeling.")
    
    with col2:
        if st.button("Train Models"):
            with st.spinner("Training models... This may take a while."):
                # Ensure data is prepared
                if not hasattr(processor, 'X_train'):
                    processor.preprocess()
                    processor.prepare_model_data()
                
                # Train models
                if processor.train_models():
                    st.success(f"Models trained successfully! Best model: {processor.best_model}")
                else:
                    st.error("Failed to train models.")
    
    # Model evaluation
    st.subheader("Model Evaluation")
    
    # Check if models are trained
    if not hasattr(processor, 'trained_models') or not processor.trained_models:
        st.warning("No models available. Please train models first.")
        return
    
    model_name = st.selectbox("Select model to evaluate", list(processor.trained_models.keys()))
    
    evaluations = processor.get_model_evaluations()
    
    if model_name in evaluations:
        st.write(f"{model_name.replace('_', ' ').title()} Evaluation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display accuracy with a larger font size and better formatting
            accuracy_value = evaluations[model_name]['accuracy'] * 100  # Convert to percentage
            st.markdown(f"""
            <div style="text-align: center; padding: 10px; background-color: rgba(0, 128, 0, 0.1); border-radius: 5px;">
                <h3 style="margin-bottom: 5px;">Model Accuracy</h3>
                <p style="font-size: 28px; font-weight: bold; color: #0066cc; margin: 0;">{accuracy_value:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Confusion matrix
            st.write("Confusion Matrix")
            cm = evaluations[model_name]['confusion_matrix']
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.title(f"Confusion Matrix - {model_name}")
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            st.pyplot(fig)
            plt.close(fig)
        
        with col2:
            st.write("Classification Report")
            report = evaluations[model_name]['classification_report']
            
            # Convert to dataframe for better display
            report_df = pd.DataFrame(report).transpose()
            # Drop support column for cleaner display
            if 'support' in report_df.columns:
                report_df = report_df.drop('support', axis=1)
            st.dataframe(report_df)
            
            # Feature importance for Random Forest
            if model_name == 'Random Forest':
                st.write("Feature Importance")
                fig = processor.get_feature_importance_plot()
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)

def render_clustering_tab(processor):
    """Render clustering section"""
    st.header("Data Clustering")
    
    if not processor.is_data_loaded:
        st.warning("Please upload data in the Data Upload tab first.")
        return
    
    # Ensure data is preprocessed
    if 'Engagement_Rate' not in processor.df.columns:
        if st.button("Preprocess Data for Clustering"):
            processor.preprocess()
            st.success("Data preprocessed successfully!")
    
    # Clustering algorithm selection
    algorithm = st.selectbox(
        "Select clustering algorithm",
        ["KMeans", "DBSCAN", "Agglomerative"],
        format_func=lambda x: {
            "KMeans": "K-Means Clustering", 
            "DBSCAN": "DBSCAN (Density-Based)", 
            "Agglomerative": "Agglomerative Hierarchical"
        }[x]
    )
    
    # Parameters based on algorithm
    if algorithm == "KMeans":
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
        params = {"n_clusters": n_clusters}
    elif algorithm == "DBSCAN":
        eps = st.slider("Epsilon (neighborhood size)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        min_samples = st.slider("Minimum samples in neighborhood", min_value=2, max_value=10, value=5)
        params = {"eps": eps, "min_samples": min_samples}
    else:  # Agglomerative
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)
        params = {"n_clusters": n_clusters}
    
    # Run clustering
    if st.button("Perform Clustering"):
        with st.spinner(f"Performing {algorithm} clustering..."):
            # Perform clustering
            if processor.perform_clustering(algorithm=algorithm.lower(), **params):
                st.success("Clustering completed!")
                
                # Display results
                fig = processor.get_clustering_plot(algorithm=algorithm.lower())
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show cluster distribution
                    if algorithm.lower() in processor.clustering_models:
                        labels = processor.clustering_models[algorithm.lower()]['labels']
                        unique_labels = np.unique(labels)
                        
                        # Count samples in each cluster
                        cluster_counts = pd.Series(labels).value_counts().sort_index()
                        
                        # For DBSCAN, cluster -1 is noise
                        if algorithm.lower() == 'dbscan' and -1 in unique_labels:
                            st.info(f"Number of noise points: {cluster_counts.get(-1, 0)} ({cluster_counts.get(-1, 0)/len(labels)*100:.1f}%)")
                            cluster_counts = cluster_counts[cluster_counts.index != -1]
                        
                        # Display cluster counts
                        st.write("Cluster Distribution:")
                        st.dataframe(pd.DataFrame({
                            'Cluster': cluster_counts.index,
                            'Count': cluster_counts.values,
                            'Percentage': [f"{count/len(labels)*100:.1f}%" for count in cluster_counts.values]
                        }))
                else:
                    st.error("Failed to generate clustering visualization.")
            else:
                st.error("Failed to perform clustering.")

def render_forecasting_tab(processor):
    """Render forecasting section"""
    st.header("Engagement Forecasting")
    
    if not processor.is_data_loaded:
        st.warning("Please upload data in the Data Upload tab first.")
        return
    
    # Check if timestamp column exists
    if 'Timestamp' not in processor.df.columns:
        st.warning("Timestamp column not found in the dataset. Forecasting requires time series data.")
        return
    
    # Ensure data is preprocessed
    if 'Engagement_Rate' not in processor.df.columns:
        if st.button("Preprocess Data for Forecasting"):
            processor.preprocess()
            st.success("Data preprocessed successfully!")
    
    st.write("Forecast Engagement Rate using Prophet time series model")
    
    if st.button("Generate Forecast"):
        with st.spinner("Generating forecast..."):
            # Prepare data and train forecasting model
            if processor.prepare_forecasting_data():
                st.success("Forecast generated successfully!")
                
                # Display forecast plot
                fig = processor.get_forecast_plot()
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show forecast components if available
                    if processor.prophet_model:
                        st.subheader("Forecast Components")
                        
                        try:
                            # Plot components
                            fig_comp = processor.prophet_model.plot_components(processor.forecast_data)
                            st.pyplot(fig_comp)
                            plt.close(fig_comp)
                        except:
                            st.warning("Could not generate components plot.")
                    
                    # Show forecast data table
                    if processor.forecast_data is not None:
                        st.subheader("Forecast Values")
                        
                        # Get future dates only
                        forecast_future = processor.forecast_data[
                            processor.forecast_data['ds'] > datetime.now()
                        ]
                        
                        # Sample data for display (e.g., monthly averages)
                        forecast_future['month'] = forecast_future['ds'].dt.to_period('M')
                        monthly_forecast = forecast_future.groupby('month').agg({
                            'ds': 'first',
                            'yhat': 'mean',
                            'yhat_lower': 'mean',
                            'yhat_upper': 'mean'
                        })
                        
                        # Format for display
                        display_df = pd.DataFrame({
                            'Month': monthly_forecast['ds'].dt.strftime('%b %Y'),
                            'Predicted Engagement': monthly_forecast['yhat'].round(2),
                            'Lower Bound': monthly_forecast['yhat_lower'].round(2),
                            'Upper Bound': monthly_forecast['yhat_upper'].round(2)
                        })
                        
                        st.dataframe(display_df)
                else:
                    st.error("Failed to create forecast plot.")
            else:
                st.error("Failed to prepare forecasting data.")

def render_yearly_sentiment_tab(processor):
    """Render yearly sentiment prediction section"""
    
    st.header("Yearly Sentiment Analysis")
    
    if not processor.is_data_loaded:
        st.warning("Please upload data in the Data Upload tab first.")
        return
        
    # Check if we can perform yearly sentiment analysis
    if 'Timestamp' not in processor.df.columns or 'Sentiment' not in processor.df.columns:
        st.error("Data must contain Timestamp and Sentiment columns for yearly sentiment analysis.")
        return
        
    # Prepare yearly sentiment data if not already done
    if not hasattr(processor, 'yearly_stats') or not processor.yearly_stats:
        if st.button("Analyze Yearly Sentiment", key="yearly_sentiment_analyze_btn"):
            with st.spinner("Analyzing yearly sentiment patterns..."):
                processor.analyze_yearly_sentiment()
                st.success("Yearly sentiment analysis completed!")
        else:
            st.info("Click the button above to analyze sentiment trends by year.")
            return
    
    # Get available years
    if hasattr(processor, 'yearly_stats') and 'counts' in processor.yearly_stats:
        available_years = processor.yearly_stats['counts'].index.tolist()
        
        if available_years:
            # Yearly stats visualization
            st.subheader("Historical Sentiment Trends")
            
            # Show yearly sentiment distribution plot
            yearly_plot = processor.get_yearly_sentiment_plot()
            if yearly_plot:
                st.pyplot(yearly_plot)
                plt.close(yearly_plot)
            else:
                st.error("Failed to create yearly sentiment plot.")
            
            # Year prediction
            st.subheader("Sentiment Prediction")
            
            # Get min and max years from available data
            min_year = min(available_years)
            max_year = max(available_years)
            
            # Let user select a year to predict
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Allow predicting years outside available range but within reasonable bounds
                prediction_year = st.number_input(
                    "Select year to predict",
                    min_value=min_year - 10,  # Allow some historical prediction
                    max_value=2050,          # Future prediction up to 2050
                    value=max_year + 1,      # Default to next year
                    step=1
                )
            
            with col2:
                st.write("Historical years in data:")
                st.write(f"From {min_year} to {max_year}")
                
                # Show hint
                if prediction_year in available_years:
                    st.info("This year exists in historical data, so prediction will be influenced by historical values.")
                else:
                    if prediction_year < min_year:
                        st.info("You're predicting a year before your historical data. Model will extrapolate backwards.")
                    elif prediction_year > max_year:
                        st.info("You're predicting a future year. Model will extrapolate based on historical trends.")
            
            # Make prediction when user clicks button
            if st.button("Predict Sentiment", key="yearly_predict_sentiment_btn"):
                with st.spinner(f"Predicting sentiment for {prediction_year}..."):
                    result = processor.predict_yearly_sentiment(prediction_year)
                    
                    if result:
                        # Get results
                        sentiment = result['sentiment']
                        probabilities = result['probabilities']
                        classes = result.get('classes', [])
                        
                        # Create DataFrame for display - ensure arrays are the same length
                        sentiments = classes
                        probs = probabilities
                        
                        # Convert any numpy arrays to lists for easier processing
                        if hasattr(sentiments, 'tolist'):
                            sentiments = sentiments.tolist()
                        if hasattr(probs, 'tolist'):
                            probs = probs.tolist()
                            
                        # Make sure arrays have the same length
                        min_len = min(len(sentiments), len(probs))
                        
                        # Create dataframe with properly sized arrays
                        prob_df = pd.DataFrame({
                            'Sentiment': sentiments[:min_len],
                            'Probability': probs[:min_len]
                        })
                        
                        # Find the highest probability sentiment from the DataFrame
                        highest_prob_idx = prob_df['Probability'].argmax()
                        highest_prob_sentiment = prob_df.iloc[highest_prob_idx]['Sentiment']
                        
                        # Get historical sentiment for comparison
                        historical_sentiment = None
                        trend_description = None
                        
                        # Check if we have historical data for this year
                        if prediction_year in available_years:
                            # Get dominant sentiment for this year from historical data
                            if 'dominant' in processor.yearly_stats:
                                if prediction_year in processor.yearly_stats['dominant'].index:
                                    historical_sentiment = processor.yearly_stats['dominant'][prediction_year]
                        
                        # Generate trend description (comparison with adjacent years)
                        if prediction_year > min_year and prediction_year < max_year:
                            # For years within our dataset, describe change from previous to next
                            prev_sentiment = processor.yearly_stats['dominant'].get(prediction_year - 1, None)
                            next_sentiment = processor.yearly_stats['dominant'].get(prediction_year + 1, None)
                            
                            if prev_sentiment and next_sentiment:
                                if prev_sentiment == highest_prob_sentiment == next_sentiment:
                                    trend_description = f"Continuing {highest_prob_sentiment} trend from {prediction_year-1} to {prediction_year+1}"
                                elif prev_sentiment == highest_prob_sentiment != next_sentiment:
                                    trend_description = f"Transitioning from {highest_prob_sentiment} to {next_sentiment} after {prediction_year}"
                                elif prev_sentiment != highest_prob_sentiment == next_sentiment:
                                    trend_description = f"Transitioning from {prev_sentiment} to {highest_prob_sentiment} at {prediction_year}"
                                else:
                                    trend_description = f"Brief {highest_prob_sentiment} sentiment, between {prev_sentiment} ({prediction_year-1}) and {next_sentiment} ({prediction_year+1})"
                        elif prediction_year <= min_year:
                            # For years before our dataset, describe future trend
                            future_sentiment = processor.yearly_stats['dominant'].get(min_year, None)
                            if future_sentiment:
                                if highest_prob_sentiment == future_sentiment:
                                    trend_description = f"Early {highest_prob_sentiment} sentiment, continuing into {min_year}"
                                else:
                                    trend_description = f"Early {highest_prob_sentiment} sentiment, changing to {future_sentiment} by {min_year}"
                        elif prediction_year >= max_year:
                            # For future years, describe trend from past
                            past_sentiment = processor.yearly_stats['dominant'].get(max_year, None)
                            if past_sentiment:
                                if highest_prob_sentiment == past_sentiment:
                                    trend_description = f"Continuation of {highest_prob_sentiment} sentiment from {max_year}"
                                else:
                                    trend_description = f"Shift from {past_sentiment} in {max_year} to {highest_prob_sentiment}"
                        
                        # Check if response has historical information from the model
                        model_historical = None
                        if 'historical' in result:
                            model_historical = result['historical']
                            
                        # Combine model historical with the directly retrieved historical sentiment
                        if model_historical and not historical_sentiment:
                            historical_sentiment = model_historical
                        
                        # Compare with historical data when available and decide how to display
                        display_sentiment = highest_prob_sentiment
                        
                        # Create appropriate display based on historical context
                        if historical_sentiment:
                            # If we have historical data for this year
                            if historical_sentiment == highest_prob_sentiment:
                                # Model agrees with historical data
                                st.markdown(f"### Predicted Sentiment: **{highest_prob_sentiment}**")
                                if trend_description:
                                    st.markdown(f"##### ({trend_description})")
                                st.success(f"✓ This prediction matches historical records for {prediction_year}")
                            else:
                                # Model disagrees with historical data
                                if trend_description:
                                    # Show trend with historical context
                                    st.markdown(f"### Predicted Sentiment: **{highest_prob_sentiment}**")
                                    st.markdown(f"##### ({trend_description})")
                                    st.warning(f"⚠️ **Historical vs Model Difference**: While the model predicts {highest_prob_sentiment}, historical data shows this year was actually **{historical_sentiment}**. The model is showing what might be expected based on overall patterns, but actual data reflects the real events from that year.")
                                else:
                                    # Just show historical context
                                    st.markdown(f"### Predicted Sentiment: **{highest_prob_sentiment}**")
                                    st.warning(f"⚠️ **Historical vs Model Difference**: While the model predicts {highest_prob_sentiment}, historical data shows this year was actually **{historical_sentiment}**.")
                                    
                                # Add a note that probabilities reflect the historical contrast
                                st.info("Note: The probabilities below reflect both the model's prediction and historical record weighting.")
                        else:
                            # No historical data, just show the prediction
                            if trend_description:
                                st.markdown(f"### Predicted Sentiment: **{highest_prob_sentiment}**")
                                st.markdown(f"##### ({trend_description})")
                            else:
                                st.markdown(f"### Predicted Sentiment: **{highest_prob_sentiment}**")
                        
                        # Sort by probability for better display
                        prob_df = prob_df.sort_values('Probability', ascending=False)
                        
                        # Format probabilities as percentages
                        prob_df['Probability %'] = prob_df['Probability'].apply(lambda x: f"{x*100:.4f}%")
                        
                        # Display the dataframe
                        st.dataframe(prob_df[['Sentiment', 'Probability', 'Probability %']])
                    else:
                        st.error("Failed to predict sentiment. Make sure you've analyzed yearly sentiment first.")
            
            # Explanation of the model
            with st.expander("How does the prediction work?"):
                st.markdown("""
                ### Yearly Sentiment Prediction Model
                
                Our model analyzes historical sentiment patterns and uses machine learning to predict sentiment for any given year. The prediction involves:
                
                1. **Historical Analysis**: We analyze the sentiment distribution across different years from your dataset.
                
                2. **Model Training**: A machine learning model is trained on the yearly patterns to identify trends and cycles.
                
                3. **Weighted Prediction**: For years with historical data, the prediction combines:
                   - 80% weight to the actual historical sentiment
                   - 20% weight to what the model would predict based on patterns
                
                4. **Trend Identification**: The system identifies sentiment transitions between years, helping you understand how sentiment evolves over time.
                
                5. **Confidence Levels**: Probability scores show the confidence of each sentiment category prediction.
                
                This approach ensures predictions respect your historical data while still showing the underlying patterns discovered by the model.
                """)
        else:
            st.warning("No yearly data available. Make sure your dataset has timestamp information.")
    else:
        st.warning("Please analyze yearly sentiment first to see trends and make predictions.")
        
def render_hypothesis_testing_tab(processor):
    """Render hypothesis testing section"""
    st.header("Non-Parametric Hypothesis Testing")
    
    if not processor.is_data_loaded:
        st.warning("Please upload data in the Data Upload tab first.")
        return
    
    # Make sure data is preprocessed - do it automatically
    if 'Engagement_Rate' not in processor.df.columns:
        with st.spinner("Preprocessing data for hypothesis testing..."):
            processor.preprocess()
            st.success("Data preprocessed successfully!")
    
    # Get the available numeric columns for testing
    numeric_cols = processor.df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = processor.df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Automatically run all tests
    
    # 1. SIGN TEST
    st.subheader("Sign Test Results")
    st.markdown("""
    **Sign Test**: A non-parametric test that assesses whether there's a difference in the median of two paired samples. 
    It only considers the sign of the differences (positive, negative, or zero) without regard to the magnitude.
    """)
    
    with st.spinner("Running Sign Test..."):
        # Determine which columns to use - use Likes and Retweets if available
        likes_col = "Likes" if "Likes" in numeric_cols else numeric_cols[0]
        retweets_col = "Retweets" if "Retweets" in numeric_cols else numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0] 
        
        # Run sign test with these columns
        sign_result = processor.perform_hypothesis_test(
            test_type="sign",
            column1=likes_col,
            column2=retweets_col
        )
        
        if 'error' in sign_result:
            st.error(f"Error in Sign Test: {sign_result['error']}")
        else:
            # Display basic info
            st.info(f"Comparing: {likes_col} vs {retweets_col}")
            
            # Create result summary
            result_summary = pd.DataFrame({
                'Metric': ['Test Type', 'P-Value', 'Significant (α=0.05)'],
                'Value': [
                    str(sign_result['test_type']),
                    f"{sign_result['p_value']:.6f}",
                    "Yes ✓" if sign_result['significant'] else "No ✗"
                ]
            })
            
            # Add test-specific metrics
            additional_metrics = pd.DataFrame({
                'Metric': ['Positive Count', 'Negative Count', 'Zero Count'],
                'Value': [
                    str(sign_result['positive_count']),
                    str(sign_result['negative_count']),
                    str(sign_result['zero_count'])
                ]
            })
            result_summary = pd.concat([result_summary, additional_metrics])
            
            # Display summary and interpretation
            col1, col2 = st.columns([1, 1])
            with col1:
                st.dataframe(result_summary)
            
            with col2:
                if sign_result['significant']:
                    st.markdown(f"**Conclusion**: There is a statistically significant difference between {likes_col} and {retweets_col} (p < 0.05).")
                else:
                    st.markdown(f"**Conclusion**: There is no statistically significant difference between {likes_col} and {retweets_col} (p ≥ 0.05).")
            
            # Display plot
            st.subheader("Sign Test Visualization")
            plot = processor.get_hypothesis_plot(sign_result)
            if plot:
                st.pyplot(plot)
                plt.close(plot)
    
    # 2. WILCOXON TEST
    st.markdown("---")
    st.subheader("Wilcoxon Signed-Rank Test Results")
    st.markdown("""
    **Wilcoxon Signed-Rank Test**: A non-parametric test for comparing two related samples. 
    Unlike the Sign Test, it considers both the sign and magnitude of the differences.
    """)
    
    with st.spinner("Running Wilcoxon Signed-Rank Test..."):
        # Determine which columns to use - use Engagement_Rate and Hashtag_Count if available
        col1 = "Engagement_Rate" if "Engagement_Rate" in numeric_cols else numeric_cols[0]
        col2 = "Hashtag_Count" if "Hashtag_Count" in numeric_cols else numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        
        # Run wilcoxon test with these columns
        wilcoxon_result = processor.perform_hypothesis_test(
            test_type="wilcoxon",
            column1=col1,
            column2=col2
        )
        
        if 'error' in wilcoxon_result:
            st.error(f"Error in Wilcoxon Test: {wilcoxon_result['error']}")
        else:
            # Display basic info
            st.info(f"Comparing: {col1} vs {col2}")
            
            # Create result summary
            result_summary = pd.DataFrame({
                'Metric': ['Test Type', 'P-Value', 'Test Statistic', 'Significant (α=0.05)'],
                'Value': [
                    str(wilcoxon_result['test_type']),
                    f"{wilcoxon_result['p_value']:.6f}",
                    f"{wilcoxon_result['statistic']:.4f}",
                    "Yes ✓" if wilcoxon_result['significant'] else "No ✗"
                ]
            })
            
            # Display summary and interpretation
            col1_ui, col2_ui = st.columns([1, 1])
            with col1_ui:
                st.dataframe(result_summary)
            
            with col2_ui:
                if wilcoxon_result['significant']:
                    st.markdown(f"**Conclusion**: There is a statistically significant difference between {col1} and {col2} (p < 0.05).")
                else:
                    st.markdown(f"**Conclusion**: There is no statistically significant difference between {col1} and {col2} (p ≥ 0.05).")
            
            # Display plot
            st.subheader("Wilcoxon Test Visualization")
            plot = processor.get_hypothesis_plot(wilcoxon_result)
            if plot:
                st.pyplot(plot)
                plt.close(plot)
    
    # 3. MANN-WHITNEY U TEST
    st.markdown("---")
    st.subheader("Mann-Whitney U Test Results")
    st.markdown("""
    **Mann-Whitney U Test**: A non-parametric test for comparing two independent samples. 
    This test doesn't assume the data follows a normal distribution.
    """)
    
    # Find categorical column for grouping
    platform_col = None
    if 'Platform' in categorical_cols:
        platform_col = 'Platform'
    elif len(categorical_cols) > 0:
        platform_col = categorical_cols[0]
    
    if platform_col and platform_col in processor.df.columns:
        with st.spinner("Running Mann-Whitney U Test..."):
            # Get unique values for platform
            group_values = processor.df[platform_col].unique().tolist()
            
            if len(group_values) >= 2:
                # Use engagement rate for comparison if available
                metric_col = "Engagement_Rate" if "Engagement_Rate" in numeric_cols else numeric_cols[0]
                group1 = group_values[0]
                group2 = group_values[1]
                
                # Run Mann-Whitney test
                mannwhitney_result = processor.perform_hypothesis_test(
                    test_type="mannwhitney",
                    column1=metric_col,
                    group_column=platform_col,
                    group1=group1,
                    group2=group2
                )
                
                if 'error' in mannwhitney_result:
                    st.error(f"Error in Mann-Whitney Test: {mannwhitney_result['error']}")
                else:
                    # Display basic info
                    st.info(f"Comparing: {metric_col} between {platform_col}={group1} and {platform_col}={group2}")
                    
                    # Create result summary
                    result_summary = pd.DataFrame({
                        'Metric': ['Test Type', 'P-Value', 'Test Statistic', 'Significant (α=0.05)'],
                        'Value': [
                            str(mannwhitney_result['test_type']),
                            f"{mannwhitney_result['p_value']:.6f}",
                            f"{mannwhitney_result['statistic']:.4f}",
                            "Yes ✓" if mannwhitney_result['significant'] else "No ✗"
                        ]
                    })
                    
                    # Add group specific info
                    additional_metrics = pd.DataFrame({
                        'Metric': [f'Group 1: {group1} (n)', f'Group 2: {group2} (n)'],
                        'Value': [
                            str(mannwhitney_result['group1_size']),
                            str(mannwhitney_result['group2_size'])
                        ]
                    })
                    result_summary = pd.concat([result_summary, additional_metrics])
                    
                    # Display summary and interpretation
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.dataframe(result_summary)
                    
                    with col2:
                        if mannwhitney_result['significant']:
                            st.markdown(f"**Conclusion**: There is a statistically significant difference in {metric_col} between {group1} and {group2} groups (p < 0.05).")
                        else:
                            st.markdown(f"**Conclusion**: There is no statistically significant difference in {metric_col} between {group1} and {group2} groups (p ≥ 0.05).")
                    
                    # Display plot
                    st.subheader("Mann-Whitney Test Visualization")
                    plot = processor.get_hypothesis_plot(mannwhitney_result)
                    if plot:
                        st.pyplot(plot)
                        plt.close(plot)
            else:
                st.info(f"Mann-Whitney U Test requires at least two groups in the categorical column '{platform_col}', but only {len(group_values)} unique values were found.")
    else:
        st.info("Mann-Whitney U Test requires a categorical column for grouping. No suitable categorical column was found in the dataset.")
            
    # Add explanation section at the bottom
    with st.expander("📚 About Non-Parametric Tests"):
        st.markdown("""
        ### Non-Parametric Hypothesis Tests
        
        Non-parametric tests are statistical methods that don't assume the data follows a specific distribution (like normal distribution). They're useful when:
        
        - Your data doesn't meet the assumptions of parametric tests
        - You have ordinal data or data with outliers
        - Your sample size is small
        
        ### Tests Available in This App:
        
        1. **Sign Test**: Simplest non-parametric test that only looks at whether one value is greater than, less than, or equal to another value.
           - Used for paired data
           - Only considers the sign of the difference, not the magnitude
           - Less powerful than Wilcoxon, but has fewer assumptions
        
        2. **Wilcoxon Signed-Rank Test**: Non-parametric alternative to the paired t-test.
           - Used for paired data
           - Takes into account both the sign and magnitude of differences
           - More powerful than the Sign Test
        
        3. **Mann-Whitney U Test**: Non-parametric alternative to the independent t-test.
           - Used for independent samples
           - Tests whether two samples come from the same distribution
           - Good for comparing groups when data isn't normally distributed
        
        ### Interpreting Results:
        
        - **p-value < 0.05**: Reject the null hypothesis. There is a statistically significant difference.
        - **p-value ≥ 0.05**: Fail to reject the null hypothesis. There is not enough evidence to claim a significant difference.
        """)
        
    # Add reference section
    with st.expander("🔍 References & Further Reading"):
        st.markdown("""
        - [Sign Test - Wikipedia](https://en.wikipedia.org/wiki/Sign_test)
        - [Wilcoxon Signed-Rank Test - Wikipedia](https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test)
        - [Mann-Whitney U Test - Wikipedia](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)
        - [SciPy Documentation for Statistical Tests](https://docs.scipy.org/doc/scipy/reference/stats.html)
        """)
        
    # Add tips for choosing tests
    with st.expander("💡 Tips for Choosing the Right Test"):
        st.markdown("""
        ### Which test should I use?
        
        - **Sign Test**: When you only care about which value is typically higher, not by how much.
          - Example: Are engagement rates consistently higher on weekends than weekdays?
        
        - **Wilcoxon Signed-Rank Test**: When you want to compare two related samples and consider the magnitude of differences.
          - Example: Did a specific change to your content strategy result in higher engagement?
        
        - **Mann-Whitney U Test**: When you want to compare independent groups without assuming normal distribution.
          - Example: Do different platforms have different average engagement rates?
        
        ### Sample Size Considerations:
        
        - For very small samples (n < 10), the Sign Test might be your only option
        - For moderate samples (10 ≤ n < 30), any of these non-parametric tests are appropriate
        - For larger samples (n ≥ 30), you might consider parametric tests if other assumptions are met
        """)
    
    # Check if necessary columns exist
    if 'Sentiment' not in processor.df.columns:
        st.warning("Sentiment column not found in the dataset.")
        return
    
    if 'Timestamp' not in processor.df.columns and 'Year' not in processor.df.columns:
        st.warning("Timestamp or Year column not found. Yearly sentiment analysis requires date information.")
        return
    
    st.write("Analyze and predict sentiment trends across years")
    
    # Analyze yearly sentiment
    if st.button("Analyze Yearly Sentiment", key="hypothesis_yearly_sentiment_btn"):
        with st.spinner("Analyzing sentiment by year..."):
            # Ensure data is preprocessed
            if 'Year' not in processor.df.columns:
                processor.preprocess()
            
            # Perform yearly sentiment analysis
            if processor.analyze_yearly_sentiment():
                st.success("Yearly sentiment analysis completed!")
                
                # Display distribution plot
                fig = processor.get_yearly_sentiment_plot()
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Show yearly statistics
                    if processor.yearly_stats and 'dominant' in processor.yearly_stats:
                        st.subheader("Dominant Sentiment by Year")
                        
                        dominant = processor.yearly_stats['dominant']
                        counts = processor.yearly_stats['counts']
                        
                        # Create display dataframe
                        display_data = []
                        for year, sentiment in dominant.items():
                            year_counts = counts.loc[year]
                            total = year_counts.sum()
                            
                            # Calculate percentages
                            percentages = (year_counts / total * 100).round(1)
                            
                            # Build row
                            row = {
                                'Year': year,
                                'Dominant Sentiment': sentiment,
                                'Total Posts': total
                            }
                            
                            # Add percentages for each sentiment
                            for sent, pct in percentages.items():
                                row[f'{sent} (%)'] = pct
                            
                            display_data.append(row)
                        
                        display_df = pd.DataFrame(display_data)
                        st.dataframe(display_df)
                else:
                    st.warning("Could not generate sentiment distribution plot.")
            else:
                st.error("Failed to analyze yearly sentiment.")
    
    # Predict sentiment for a specific year
    st.subheader("Predict Sentiment for a Specific Year")
    
    # Check if yearly sentiment model is trained
    if not hasattr(processor, 'yearly_sentiment_model') or processor.yearly_sentiment_model is None:
        st.info("Please run 'Analyze Yearly Sentiment' first to train the prediction model.")
    else:
        # Year selection
        min_year = min(processor.yearly_stats['years'])
        max_year = 2050  # Allow prediction up to 2050
        
        prediction_year = st.selectbox(
            "Select year for prediction", 
            range(min_year, max_year + 1),
            index=5  # Default to a few years in the future
        )
        
        if st.button("Predict Sentiment", key="hypothesis_predict_sentiment_btn"):
            with st.spinner(f"Predicting sentiment for {prediction_year}..."):
                result = processor.predict_yearly_sentiment(prediction_year)
                
                if result:
                    st.success(f"Prediction for {prediction_year} completed!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Create dataframe for probabilities - ensure arrays are the same length
                        sentiments = result.get('classes', [])
                        probabilities = result.get('probabilities', [])
                        
                        # Convert any numpy arrays to lists for easier processing
                        if hasattr(sentiments, 'tolist'):
                            sentiments = sentiments.tolist()
                        if hasattr(probabilities, 'tolist'):
                            probabilities = probabilities.tolist()
                        
                        # Ensure both arrays exist and have content
                        if not sentiments or not probabilities:
                            st.error("Missing sentiment classes or probabilities in result")
                            return
                            
                        # Make sure arrays have the same length
                        min_len = min(len(sentiments), len(probabilities))
                        
                        # Create dataframe with proper types
                        prob_df = pd.DataFrame({
                            'Sentiment': sentiments[:min_len],
                            'Probability': probabilities[:min_len]
                        })
                        
                        # Recompute highest probability sentiment here to ensure it's consistent with display
                        highest_prob_idx = prob_df['Probability'].argmax()  
                        highest_prob_sentiment = prob_df.iloc[highest_prob_idx]['Sentiment']
                        
                        # Display the sentiment with the highest probability and context
                        # Find top two sentiments for trend analysis
                        top_sentiments = prob_df.head(2)
                        
                        # Get historical context from the yearly_stats
                        historical_sentiment = None
                        if prediction_year in processor.yearly_stats.get('dominant', {}).keys():
                            historical_sentiment = processor.yearly_stats['dominant'][prediction_year]
                        
                        # Find trend direction based on history and top probabilities
                        trend_description = ""
                        if len(top_sentiments) >= 2:
                            # Calculate probability ratio to determine confidence in the transition
                            prob_ratio = top_sentiments.iloc[0]['Probability'] / (top_sentiments.iloc[1]['Probability'] + 0.0001)
                            
                            # If ratio is small enough, it's a borderline case between top two sentiments
                            if prob_ratio < 10:  # Less than 10x difference suggests uncertainty
                                trend_description = f"trending between {top_sentiments.iloc[0]['Sentiment']} and {top_sentiments.iloc[1]['Sentiment']}"
                        
                        # Check if response has historical information from the model
                        model_historical = None
                        if 'historical' in result:
                            model_historical = result['historical']
                            
                        # Combine model historical with the directly retrieved historical sentiment
                        if model_historical and not historical_sentiment:
                            historical_sentiment = model_historical
                        
                        # Compare with historical data when available and decide how to display
                        display_sentiment = highest_prob_sentiment
                        
                        # Create appropriate display based on historical context
                        if historical_sentiment:
                            # If we have historical data for this year
                            if historical_sentiment == highest_prob_sentiment:
                                # Model agrees with historical data
                                st.markdown(f"### Predicted Sentiment: **{highest_prob_sentiment}**")
                                if trend_description:
                                    st.markdown(f"##### ({trend_description})")
                                st.success(f"✓ This prediction matches historical records for {prediction_year}")
                            else:
                                # Model disagrees with historical data
                                if trend_description:
                                    # Show trend with historical context
                                    st.markdown(f"### Predicted Sentiment: **{highest_prob_sentiment}**")
                                    st.markdown(f"##### ({trend_description})")
                                    st.warning(f"⚠️ **Historical vs Model Difference**: While the model predicts {highest_prob_sentiment}, historical data shows this year was actually **{historical_sentiment}**. The model is showing what might be expected based on overall patterns, but actual data reflects the real events from that year.")
                                else:
                                    # Just show historical context
                                    st.markdown(f"### Predicted Sentiment: **{highest_prob_sentiment}**")
                                    st.warning(f"⚠️ **Historical vs Model Difference**: While the model predicts {highest_prob_sentiment}, historical data shows this year was actually **{historical_sentiment}**.")
                                    
                                # Add a note that probabilities reflect the historical contrast
                                st.info("Note: The probabilities below reflect both the model's prediction and historical record weighting.")
                        else:
                            # No historical data, just show the prediction
                            if trend_description:
                                st.markdown(f"### Predicted Sentiment: **{highest_prob_sentiment}**")
                                st.markdown(f"##### ({trend_description})")
                            else:
                                st.markdown(f"### Predicted Sentiment: **{highest_prob_sentiment}**")
                        
                        # Sort by probability for better display
                        prob_df = prob_df.sort_values('Probability', ascending=False)
                        
                        # Format probabilities as percentages
                        prob_df['Probability %'] = prob_df['Probability'].apply(lambda x: f"{x*100:.4f}%")
                        
                        # Display the dataframe
                        st.dataframe(prob_df[['Sentiment', 'Probability', 'Probability %']])
                    
                    with col2:
                        # Create bar chart for probabilities - use the sorted dataframe
                        fig, ax = plt.subplots(figsize=(8, 4))
                        
                        # Define consistent colors for each sentiment
                        colors = {'Negative': '#ff9999', 'Neutral': '#66b3ff', 'Positive': '#99ff99'}
                        default_colors = ['#ff9999', '#66b3ff', '#99ff99'] 
                        
                        # Get colors for each sentiment, fallback to default if not found
                        bar_colors = [colors.get(sentiment, default_colors[i % len(default_colors)]) 
                                      for i, sentiment in enumerate(prob_df['Sentiment'])]
                        
                        # Create the bar plot with the sorted data
                        bars = ax.bar(prob_df['Sentiment'], prob_df['Probability'], color=bar_colors)
                        
                        # Highlight the highest probability with value annotation
                        for i, bar in enumerate(bars):
                            height = bar.get_height()
                            # Format with more precision for very small values
                            if height < 0.01:
                                label = f"{height:.6f}"
                            else:
                                label = f"{height:.4f}"
                            ax.annotate(label,
                                        xy=(bar.get_x() + bar.get_width() / 2, height),
                                        xytext=(0, 3),  # 3 points vertical offset
                                        textcoords="offset points",
                                        ha='center', va='bottom',
                                        fontsize=10,
                                        fontweight='bold' if i == 0 else 'normal')
                        
                        ax.set_ylim(0, 1)
                        ax.set_title(f"Sentiment Probability for {prediction_year}")
                        ax.set_ylabel("Probability")
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Show comparison with past years if available
                    if processor.yearly_stats and 'dominant' in processor.yearly_stats:
                        st.subheader("Sentiment Trend Over Years")
                        
                        # Get historical data
                        years = list(processor.yearly_stats['dominant'].index)
                        sentiments = list(processor.yearly_stats['dominant'].values)
                        
                        # Add prediction - use the highest probability sentiment we calculated
                        if prediction_year not in years:
                            years.append(prediction_year)
                            sentiments.append(highest_prob_sentiment)  # Use the highest probability sentiment
                        
                        # Create dataframe
                        trend_df = pd.DataFrame({
                            'Year': years,
                            'Dominant Sentiment': sentiments
                        })
                        
                        # Sort by year
                        trend_df = trend_df.sort_values('Year')
                        
                        # Display as table
                        st.dataframe(trend_df)
                        
                        # Mark the predicted year
                        if prediction_year in trend_df['Year'].values:
                            st.info(f"The sentiment for {prediction_year} is a prediction, all others are historical data.")
                            
                            # Add a temporal trend analysis using past data
                            if prediction_year in years and isinstance(prediction_year, (int, float)):
                                # Find the closest years with data
                                past_years = [y for y in sorted(years) if y < prediction_year]
                                
                                if past_years:
                                    recent_past = past_years[-min(3, len(past_years)):]  # Get up to 3 most recent years
                                    past_sentiments = []
                                    
                                    for py in recent_past:
                                        idx = years.index(py)
                                        past_sentiments.append(sentiments[idx])
                                    
                                    # Analyze temporal pattern
                                    if len(set(past_sentiments)) == 1 and past_sentiments[0] != highest_prob_sentiment:
                                        # Stable past, different prediction
                                        st.warning(f"⚠️ **Trend Change Alert**: Our model predicts a shift from a stable {past_sentiments[0]} pattern to {highest_prob_sentiment}. This could indicate an emerging trend or pattern change.")
                                    
                                    elif past_sentiments and past_sentiments[-1] != highest_prob_sentiment:
                                        # Different from most recent year
                                        st.info(f"📊 **Sentiment Transition**: The prediction shows a change from {past_sentiments[-1]} to {highest_prob_sentiment}, suggesting a possible shift in sentiment patterns.")
                                    
                                    # Create a temporal trend visualization
                                    st.subheader("Sentiment Trend Visualization")
                                    
                                    # Get the years and sentiments for visualization
                                    trend_years = [int(y) for y in sorted(years)]
                                    trend_sentiments = []
                                    sentiment_values = {"Negative": -1, "Neutral": 0, "Positive": 1}
                                    
                                    for y in trend_years:
                                        if y in years:
                                            idx = years.index(y)
                                            s = sentiments[idx]
                                            trend_sentiments.append(sentiment_values.get(s, 0))
                                    
                                    # Create the trend plot
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    
                                    # Plot the sentiment values
                                    ax.plot(trend_years, trend_sentiments, marker='o', linestyle='-', color='#3366cc')
                                    
                                    # Highlight the prediction
                                    # Find the index of the prediction year in the trend_years list
                                    pred_idx = None
                                    for i, year in enumerate(trend_years):
                                        if year == int(prediction_year):
                                            pred_idx = i
                                            break
                                    
                                    if pred_idx is not None:
                                        ax.plot(trend_years[pred_idx], trend_sentiments[pred_idx], 
                                               marker='*', markersize=15, color='red')
                                    
                                    # Only add visualization elements if we found a valid prediction index
                                    if pred_idx is not None:
                                        # Add a shaded area for the prediction's uncertainty
                                        # The width of the shade depends on the probability ratio
                                        prob_range = 0.2 * (1 - top_sentiments.iloc[0]['Probability'])
                                        
                                        ax.fill_between([trend_years[pred_idx]], 
                                                       [trend_sentiments[pred_idx] - prob_range],
                                                       [trend_sentiments[pred_idx] + prob_range],
                                                       color='red', alpha=0.2)
                                        
                                        # Add annotation for prediction
                                        ax.annotate('Prediction', 
                                                  xy=(trend_years[pred_idx], trend_sentiments[pred_idx]),
                                                  xytext=(10, 0),
                                                  textcoords="offset points",
                                                  arrowprops=dict(arrowstyle="->"),
                                                  fontweight='bold')
                                    
                                    # Set y-axis ticks and labels
                                    ax.set_yticks([-1, 0, 1])
                                    ax.set_yticklabels(['Negative', 'Neutral', 'Positive'])
                                    
                                    # Set title and labels
                                    ax.set_title('Sentiment Trend Over Time')
                                    ax.set_xlabel('Year')
                                    ax.grid(True, alpha=0.3)
                                    
                                    st.pyplot(fig)
                                    plt.close(fig)
                else:
                    st.error("Failed to predict sentiment.")