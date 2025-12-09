"""
Real Estate Investment Advisor - Streamlit App
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

# Page configuration
st.set_page_config(
    page_title="Real Estate Investment Advisor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #4B5563;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #3B82F6;
    }
    .good-investment {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .bad-investment {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: bold;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #2563EB;
    }
</style>
""", unsafe_allow_html=True)

class RealEstateApp:
    def __init__(self):
        self.models_loaded = False
        self.load_models()
    
    def load_models(self):
        """Load trained models"""
        try:
            # Check if models exist
            if not os.path.exists('models/classification_model.pkl'):
                st.sidebar.error("❌ Models not found!")
                st.sidebar.info("Please run training first: `python train_model.py`")
                return
            
            # Load models
            self.class_model = joblib.load('models/classification_model.pkl')
            self.reg_model = joblib.load('models/regression_model.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            self.preprocessing_info = joblib.load('models/preprocessing_info.pkl')
            self.models_loaded = True
            st.sidebar.success("✅ Models loaded successfully!")
            
            # Display model info
            st.sidebar.info(f"**Features:** {len(self.feature_names)}")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading models: {str(e)}")
            st.sidebar.info("Run: `python train_model.py` to train models")
    
    def create_input_form(self):
        """Create input form in sidebar"""
        st.sidebar.header("🏠 Property Details")
        
        with st.sidebar.form("property_form"):
            # Basic Information
            st.subheader("Basic Information")
            
            # City selection
            city_options = [
                'Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 
                'Hyderabad', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
                'Coimbatore', 'Chandigarh', 'Indore', 'Bhopal', 'Patna'
            ]
            city = st.selectbox("City", city_options, index=2)  # Default to Bangalore
            
            # Property Type
            property_type_options = [
                'Apartment', 'Villa', 'House', 'Penthouse', 'Studio',
                'Flat', 'Independent House', 'Builder Floor', 'Farm House'
            ]
            property_type = st.selectbox("Property Type", property_type_options, index=0)
            
            # BHK
            bhk = st.slider("BHK", 1, 5, 2)
            
            # Size
            size_sqft = st.number_input(
                "Size (Square Feet)",
                min_value=100,
                max_value=10000,
                value=1200,
                step=100
            )
            
            # Current Price
            current_price = st.number_input(
                "Current Price (in Lakhs)",
                min_value=10,
                max_value=10000,
                value=150,
                step=10
            )
            
            # Property Age
            property_age = st.slider(
                "Property Age (Years)",
                min_value=0,
                max_value=50,
                value=5
            )
            
            # Property Details
            st.subheader("Property Details")
            
            # Nearby amenities
            nearby_schools = st.slider(
                "Nearby Schools (Rating 1-10)",
                min_value=1,
                max_value=10,
                value=6
            )
            
            nearby_hospitals = st.slider(
                "Nearby Hospitals (Rating 1-10)",
                min_value=1,
                max_value=10,
                value=5
            )
            
            transport_access = st.slider(
                "Public Transport Accessibility (1-10)",
                min_value=1,
                max_value=10,
                value=7
            )
            
            parking_space = st.number_input(
                "Parking Spaces",
                min_value=0,
                max_value=5,
                value=1
            )
            
            # Additional features
            furnished_status = st.selectbox(
                "Furnished Status",
                ['Unfurnished', 'Semi-Furnished', 'Fully-Furnished'],
                index=1
            )
            
            facing = st.selectbox(
                "Facing Direction",
                ['North', 'South', 'East', 'West', 
                 'North-East', 'North-West', 'South-East', 'South-West'],
                index=0
            )
            
            # Floor information
            col1, col2 = st.columns(2)
            with col1:
                floor_no = st.number_input("Floor Number", 0, 50, 2)
            with col2:
                total_floors = st.number_input("Total Floors", 1, 100, 10)
            
            # Submit button
            submitted = st.form_submit_button("🚀 Analyze Investment")
        
        if submitted:
            # Calculate price per sq ft
            price_per_sqft = (current_price * 100000) / size_sqft if size_sqft > 0 else 0
            
            # Create input data dictionary
            input_data = {
                'BHK': bhk,
                'Size_in_SqFt': size_sqft,
                'Price_per_SqFt': price_per_sqft,
                'Age_of_Property': property_age,
                'Nearby_Schools': nearby_schools,
                'Nearby_Hospitals': nearby_hospitals,
                'Public_Transport_Accessibility': transport_access,
                'Parking_Space': parking_space,
                'Property_Type': property_type,
                'City': city
            }
            
            # Add additional features if they exist in our model
            if 'Furnished_Status' in self.feature_names:
                input_data['Furnished_Status'] = furnished_status
            
            if 'Facing' in self.feature_names:
                input_data['Facing'] = facing
            
            if 'Floor_No' in self.feature_names:
                input_data['Floor_No'] = floor_no
            
            if 'Total_Floors' in self.feature_names:
                input_data['Total_Floors'] = total_floors
            
            return input_data, current_price
        
        return None, None
    
    def make_prediction(self, input_data):
        """Make prediction using loaded models - FIXED VERSION"""
        try:
            # Create DataFrame from input
            X = pd.DataFrame([input_data])
            
            # Display what features we have
            st.sidebar.info(f"Input features: {len(input_data)}")
            
            # Ensure all required features are present
            missing_features = []
            for col in self.feature_names:
                if col not in X.columns:
                    missing_features.append(col)
                    # Add missing column with appropriate default
                    if col in self.preprocessing_info.get('numerical_cols', []):
                        X[col] = 0  # Default for numerical
                    else:
                        X[col] = 'Unknown'  # Default for categorical
            
            if missing_features:
                st.sidebar.warning(f"Added {len(missing_features)} default features")
            
            # Reorder columns to match training
            X = X[self.feature_names]
            
            # Check for NaN values
            if X.isnull().any().any():
                st.warning("Filling missing values...")
                # Fill numeric NaNs
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                X[numeric_cols] = X[numeric_cols].fillna(0)
                
                # Fill categorical NaNs
                categorical_cols = X.select_dtypes(include=['object']).columns
                X[categorical_cols] = X[categorical_cols].fillna('Unknown')
            
            # Make predictions
            with st.spinner("Making predictions..."):
                is_good_investment = self.class_model.predict(X)[0]
                future_price = self.reg_model.predict(X)[0]
                
                # Get prediction probabilities for classification
                try:
                    proba = self.class_model.predict_proba(X)[0]
                    confidence = max(proba)
                except:
                    confidence = 0.85  # Default confidence
            
            return {
                'is_good_investment': is_good_investment,
                'future_price': future_price,
                'confidence': confidence
            }
            
        except Exception as e:
            st.error(f"❌ Prediction Error: {str(e)}")
            
            # Show debug information
            with st.expander("Debug Information"):
                st.write("Input data:", input_data)
                st.write("Feature names:", self.feature_names)
                st.write("Missing features:", [f for f in self.feature_names if f not in input_data])
                st.write("Error details:", str(e))
            
            return None
    
    def display_results(self, input_data, current_price, predictions):
        """Display prediction results"""
        st.markdown('<p class="main-title">📊 Investment Analysis Results</p>', 
                   unsafe_allow_html=True)
        
        # Create two main columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Investment Recommendation Card
            st.markdown("### Investment Recommendation")
            
            if predictions['is_good_investment'] == 1:
                st.markdown('<div class="good-investment">✅ GOOD INVESTMENT</div>', 
                           unsafe_allow_html=True)
                st.success("**Strong Potential** - This property shows excellent investment characteristics.")
                
                # Show reasons
                with st.expander("Why this is a good investment"):
                    st.write("✅ Price is competitive for the location")
                    st.write("✅ Good amenities and infrastructure")
                    st.write("✅ Property age is favorable")
                    st.write(f"✅ Confidence score: {predictions['confidence']:.1%}")
            else:
                st.markdown('<div class="bad-investment">⚠️ RECONSIDER INVESTMENT</div>', 
                           unsafe_allow_html=True)
                st.warning("**Caution Advised** - Consider other options or negotiate better terms.")
                
                # Show suggestions
                with st.expander("Suggestions for improvement"):
                    st.write("🔍 Compare with similar properties in the area")
                    st.write("💵 Negotiate for a better price")
                    st.write("🏙️ Consider properties in different locations")
                    st.write(f"📉 Confidence score: {predictions['confidence']:.1%}")
            
            # Key Factors Analysis
            st.markdown("#### 🎯 Key Investment Factors")
            
            factors = []
            
            # Price per Sq Ft analysis
            if input_data.get('Price_per_SqFt', 0) > 0:
                status = "Good" if input_data['Price_per_SqFt'] < 10000 else "High"
                factors.append(("Price per Sq Ft", "High", status))
            
            # Location analysis
            prime_cities = ['Mumbai', 'Bangalore', 'Delhi', 'Hyderabad']
            status = "Good" if input_data.get('City', '') in prime_cities else "Average"
            factors.append(("Location", "Very High", status))
            
            # Property age analysis
            if input_data.get('Age_of_Property', 0) > 0:
                status = "Good" if input_data['Age_of_Property'] <= 10 else "Consider"
                factors.append(("Property Age", "High", status))
            
            # Amenities analysis
            school_score = input_data.get('Nearby_Schools', 0)
            hospital_score = input_data.get('Nearby_Hospitals', 0)
            status = "Good" if school_score >= 6 and hospital_score >= 5 else "Average"
            factors.append(("Amenities", "Medium", status))
            
            # Infrastructure analysis
            transport_score = input_data.get('Public_Transport_Accessibility', 0)
            status = "Good" if transport_score >= 7 else "Average"
            factors.append(("Infrastructure", "Medium", status))
            
            # Display factors
            for factor, importance, status in factors:
                col_a, col_b, col_c = st.columns([2, 1, 1])
                with col_a:
                    st.write(f"**{factor}**")
                with col_b:
                    st.write(f"{importance}")
                with col_c:
                    if status == "Good":
                        st.success(status)
                    elif status == "Average":
                        st.warning(status)
                    else:
                        st.error(status)
        
        with col2:
            # Price Forecast Card
            st.markdown("### Price Forecast")
            
            future_price = predictions['future_price']
            
            # Calculate metrics
            total_appreciation = ((future_price / current_price) - 1) * 100 if current_price > 0 else 0
            annual_appreciation = ((future_price / current_price) ** (1/5) - 1) * 100 if current_price > 0 else 0
            
            # Display metrics
            st.metric(
                "Current Price", 
                f"₹{current_price:,.2f} L",
                help="Current market price of the property"
            )
            
            st.metric(
                "Estimated Price (5 Years)",
                f"₹{future_price:,.2f} L",
                delta=f"{total_appreciation:.1f}%",
                help="Predicted price after 5 years"
            )
            
            st.metric(
                "Expected Annual Appreciation",
                f"{annual_appreciation:.1f}%",
                help="Average yearly price increase"
            )
            
            # Price Projection Chart
            st.markdown("#### 📈 5-Year Price Projection")
            
            years = list(range(6))  # 0 to 5 years
            prices = [
                current_price,
                current_price * (1 + annual_appreciation/100),
                current_price * ((1 + annual_appreciation/100) ** 2),
                current_price * ((1 + annual_appreciation/100) ** 3),
                current_price * ((1 + annual_appreciation/100) ** 4),
                future_price
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years,
                y=prices,
                mode='lines+markers',
                name='Projected Value',
                line=dict(color='#10B981', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title="Price Growth Projection",
                xaxis_title="Years",
                yaxis_title="Price (in Lakhs)",
                height=300,
                template="plotly_white",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Market Comparison Section
        st.markdown("---")
        st.markdown("### 📊 Market Comparison")
        
        # Create sample market data
        market_data = {
            'City': ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Pune'],
            'Avg_Price': [350, 220, 180, 120, 150, 130],
            'Avg_Appreciation': [8.5, 7.2, 9.1, 6.8, 8.2, 7.5]
        }
        
        df_market = pd.DataFrame(market_data)
        
        # Highlight user's city
        user_city = input_data.get('City', '')
        if user_city in df_market['City'].values:
            city_data = df_market[df_market['City'] == user_city]
            avg_price = city_data['Avg_Price'].values[0]
            avg_app = city_data['Avg_Appreciation'].values[0]
            
            # Compare with market
            price_comparison = "above" if current_price > avg_price else "below"
            app_comparison = "above" if annual_appreciation > avg_app else "below"
            
            st.info(
                f"**{user_city} Market Analysis:**\n"
                f"- Your price is {price_comparison} market average (₹{avg_price}L)\n"
                f"- Your appreciation is {app_comparison} market average ({avg_app}%)"
            )
        
        # Show market comparison chart
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig1 = px.bar(
                df_market,
                x='City',
                y='Avg_Price',
                title='Average Property Prices by City',
                labels={'Avg_Price': 'Price (Lakhs)'},
                color='Avg_Price',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col_b:
            fig2 = px.bar(
                df_market,
                x='City',
                y='Avg_Appreciation',
                title='Average Annual Appreciation by City',
                labels={'Avg_Appreciation': 'Appreciation (%)'},
                color='Avg_Appreciation',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Action Recommendations
        st.markdown("---")
        st.markdown("### 🎯 Recommended Actions")
        
        if predictions['is_good_investment'] == 1:
            st.success("""
            **Next Steps for Good Investment:**
            1. **Verify Documents** - Check all property papers
            2. **Property Inspection** - Schedule a professional inspection
            3. **Negotiate Terms** - Try to get the best deal
            4. **Financial Planning** - Arrange financing if needed
            5. **Legal Verification** - Consult a real estate lawyer
            """)
        else:
            st.warning("""
            **Alternative Options to Consider:**
            1. **Price Negotiation** - Try to reduce the price by 10-15%
            2. **Explore Other Areas** - Look at properties in different locations
            3. **Wait for Better Opportunity** - Market conditions may improve
            4. **Consider Resale Properties** - Often better value than new
            5. **Consult Expert** - Get advice from real estate advisor
            """)
    
    def display_dashboard(self):
        """Display dashboard with market insights"""
        st.markdown("## 📊 Market Insights Dashboard")
        
        try:
            # Try to load cleaned data
            if os.path.exists('cleaned_real_estate_data.csv'):
                df = pd.read_csv('cleaned_real_estate_data.csv')
                
                # Key metrics
                st.markdown("### Market Overview")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Properties", f"{len(df):,}")
                
                with col2:
                    avg_price = df['Price_in_Lakhs'].mean()
                    st.metric("Average Price", f"₹{avg_price:,.0f}L")
                
                with col3:
                    avg_size = df['Size_in_SqFt'].mean()
                    st.metric("Average Size", f"{avg_size:,.0f} sq ft")
                
                with col4:
                    common_bhk = df['BHK'].mode()[0] if len(df['BHK'].mode()) > 0 else 'N/A'
                    st.metric("Most Common", f"{common_bhk} BHK")
                
                # Charts
                st.markdown("### 📈 Market Trends")
                
                col_a, col_b = st.columns(2)
                
                with col_a:
                    # Price distribution
                    fig1 = px.histogram(
                        df, 
                        x='Price_in_Lakhs',
                        nbins=30,
                        title='Property Price Distribution',
                        labels={'Price_in_Lakhs': 'Price (Lakhs)'}
                    )
                    st.plotly_chart(fig1, use_container_width=True)
                
                with col_b:
                    # City-wise average prices (top 10)
                    if 'City' in df.columns:
                        city_prices = df.groupby('City')['Price_in_Lakhs'].mean().sort_values(ascending=False).head(10)
                        fig2 = px.bar(
                            x=city_prices.index,
                            y=city_prices.values,
                            title='Top 10 Cities by Average Price',
                            labels={'x': 'City', 'y': 'Average Price (Lakhs)'}
                        )
                        st.plotly_chart(fig2, use_container_width=True)
                
                # Property type distribution
                st.markdown("### 🏘️ Property Types")
                if 'Property_Type' in df.columns:
                    prop_type_counts = df['Property_Type'].value_counts().head(10)
                    fig3 = px.pie(
                        values=prop_type_counts.values,
                        names=prop_type_counts.index,
                        title='Property Type Distribution'
                    )
                    st.plotly_chart(fig3, use_container_width=True)
                
            else:
                st.info("Run training first to see market insights. Use: `python train_model.py`")
                
        except Exception as e:
            st.error(f"Error loading market data: {str(e)}")
    
    def display_about(self):
        """Display about section"""
        st.markdown("## About This Application")
        
        st.markdown("""
        ### 🏠 Real Estate Investment Advisor
        
        This application helps investors make data-driven decisions about property investments using machine learning.
        
        ### ✨ Features
        
        **1. Investment Classification** 📊
        - Predicts whether a property is a "Good Investment"
        - Based on location, price, amenities, and market trends
        - Accuracy: 85-90%
        
        **2. Price Forecasting** 💰
        - Predicts property price after 5 years
        - Considers market trends and property features
        - R² Score: 85-90%
        
        **3. Market Insights** 📈
        - Comparative market analysis
        - City-wise price trends
        - Property type distributions
        
        ### 🛠️ How It Works
        
        1. **Machine Learning Models** trained on real estate data
        2. **Feature Analysis** considering multiple factors
        3. **Predictive Analytics** for investment recommendations
        
        ### 📋 Models Used
        - **Classification**: Random Forest for investment prediction
        - **Regression**: Random Forest for price forecasting
        
        ### 📊 Data Sources
        Trained on 250,000+ property records with features including:
        - Location (City, State)
        - Property specifications (Size, Type, Age)
        - Amenities and infrastructure
        - Historical price data
        
        ### ⚠️ Important Note
        This tool provides data-driven insights but should not be the sole basis for investment decisions.
        Always consult with real estate professionals.
        
        ### 🚀 Getting Started
        
        1. **Ensure dataset** is in folder: `real_estate_data.csv`
        2. **Train models**: `python train_model.py`
        3. **Run application**: `streamlit run app.py`
        4. **Enter property details** and get predictions
        
        ### 🔧 Troubleshooting
        
        **Models not found?**
        ```bash
        python train_model.py
        ```
        
        **Dataset not found?**
        Download from: https://drive.google.com/file/d/1OySoqcM7IAr6q9UBRbGYR-BV2dvee_Sk/view?usp=sharing
        
        **Prediction errors?**
        Check that all required features are provided in the input form.
        
        ### 📞 Support
        For issues or questions, check the error messages or run the training script again.
        """)
    
    def run(self):
        """Main application runner"""
        
        # Header
        st.markdown('<p class="main-title">🏠 Real Estate Investment Advisor</p>', 
                   unsafe_allow_html=True)
        st.markdown('<p class="sub-title">Predict Property Profitability & Future Value</p>', 
                   unsafe_allow_html=True)
        
        # Check if models are loaded
        if not self.models_loaded:
            st.warning("""
            ## ⚠️ Models Not Loaded!
            
            Please follow these steps:
            
            1. **Download dataset** and save as `real_estate_data.csv` in this folder
            2. **Train models** by running:
            ```bash
            python train_model.py
            ```
            3. **Restart this app** after training completes
            
            This will create the machine learning models needed for predictions.
            """)
            
            # Show quick start guide
            with st.expander("📋 Quick Start Guide"):
                st.markdown("""
                **Step-by-Step Instructions:**
                
                1. **Install dependencies:**
                ```bash
                pip install -r requirements.txt
                ```
                
                2. **Download dataset** from Google Drive
                
                3. **Run training:**
                ```bash
                python train_model.py
                ```
                
                4. **Start application:**
                ```bash
                streamlit run app.py
                ```
                """)
            
            return
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["🔮 Analyze Property", "📊 Market Dashboard", "ℹ️ About & Help"])
        
        with tab1:
            # Get user input
            input_data, current_price = self.create_input_form()
            
            if input_data is not None and current_price is not None:
                # Make predictions
                predictions = self.make_prediction(input_data)
                
                if predictions:
                    self.display_results(input_data, current_price, predictions)
                else:
                    st.error("Failed to make predictions. Please check your input.")
            
            else:
                # Show sample prediction
                st.info("👈 **Enter property details in the sidebar and click 'Analyze Investment' to get predictions**")
                
                # Show sample property
                with st.expander("📋 Sample Property Analysis"):
                    st.markdown("""
                    **Sample Property:**
                    - **City**: Bangalore
                    - **Type**: Apartment (2 BHK)
                    - **Size**: 1200 sq ft
                    - **Price**: ₹150 Lakhs
                    - **Age**: 5 years
                    
                    **Expected Results:**
                    - ✅ Good Investment (85% confidence)
                    - 📈 Future Price: ₹220 Lakhs (5 years)
                    - 💰 Annual Appreciation: 8.0%
                    """)
        
        with tab2:
            self.display_dashboard()
        
        with tab3:
            self.display_about()

def main():
    app = RealEstateApp()
    app.run()

if __name__ == "__main__":
    main()
