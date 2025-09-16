import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

# Database connection
def create_connection():
    try:
        # Update credentials as needed (replace 'admin' with the actual password and 'project' with your database name)
        engine = create_engine("mysql+mysqldb://root:admin@localhost:3306/project")
        connection = engine.connect()
        return connection
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return None

# Fetch data from database
def fetch_data(query):
    connection = create_connection()
    if connection:
        try:
            # Use pandas to read SQL directly via the SQLAlchemy connection
            df = pd.read_sql(query, connection)
            return df
        finally:
            connection.close()
    else:
        return pd.DataFrame()
# Streamlit UI
st.set_page_config(page_title="Securecheck Police Dashboard", layout="wide")

st.title("🔐 Securecheck: Police Check Post Digital Ledger")
st.markdown("Real-time monitoring and insights for law enforcement 🚔")

# Show full table
st.header("🗂️ Police Logs Overview")
query = "SELECT * FROM traffic_stop_log"
data = fetch_data(query)

data['stop_time'] = data['timestamp'].dt.time

st.dataframe(data, use_container_width=True)

# Quick Metrics
st.header("📊 Dashboard Metrics")  # [1]

col1, col2, col3, col4 = st.columns(4)  # [3]

with col1:
    total_stops = len(data)  # or data.shape
    st.metric("Total Police Stops", total_stops)  

with col2:
    arrests = data['stop_outcome'].str.contains("Arrest", case=False, na=False).sum()
    st.metric("Total Arrests", arrests) 

with col3:
    warnings = data['stop_outcome'].str.contains("Warning", case=False, na=False).sum()
    st.metric("Total Warnings", warnings) 

with col4:
    drug_related_stop = (data['drugs_related_stop'] == 1).sum()
    st.metric("Total Drug related stop", drug_related_stop)  

st.header("🗂️ Vehicle based report")
vehicle_based_report = st.selectbox(
    "Choose Visualization:",
    [
        "What are the top 10 vehicle_Number involved in drug-related stops?",
        "Which vehicles were most frequently searched?"
    ]
)
if vehicle_based_report == "What are the top 10 vehicle_Number involved in drug-related stops?":
    if not data.empty and 'vehicle_number' in data.columns and 'drugs_related_stop' in data.columns:
        drug_stops = data[data['drugs_related_stop'] == 1]

        top_vehicles = (
            drug_stops['vehicle_number']
            .value_counts()
            .head(10)
            .reset_index()
        )
        top_vehicles.columns = ['Vehicle Number', 'Stop Count']

        st.write("Top 10 vehicle numbers involved in drug-related stops:")
        st.dataframe(top_vehicles)
    else:
        st.warning("Data missing for vehicle_number or drugs_related_stop.")

elif vehicle_based_report == "Which vehicles were most frequently searched?":
    if not data.empty and 'vehicle_number' in data.columns and 'search_conducted' in data.columns:
        # Filter rows where a search was actually conducted (assuming search_conducted is 1 for yes)
        searched_vehicles = data[data['search_conducted'] == 1]

        # Count the occurrences of each vehicle_number to get most frequently searched vehicles
        top_searched_vehicles = (
            searched_vehicles['vehicle_number']
            .value_counts()
            .head(20)
            .reset_index()
        )
        top_searched_vehicles.columns = ['Vehicle Number', 'Search Count']

        st.write("Top 20 most frequently searched vehicles:")
        st.dataframe(top_searched_vehicles)
    else:
        st.warning("Data missing for vehicle_number or search_conducted.")


# Charts
# Streamlit section for Visualization Select Box
st.header("🗂️ Visual Insights")

# User selects desired visualization
visualization_option = st.selectbox(
    "Choose a visualization:",
    [
        "👨‍🦳 Highest Arrest Rate by Driver Age Group",
        "What is the gender distribution of drivers stopped in each country?",
         "Which race and gender combination has the highest search rate?"
    ]
)

# 👨‍🦳 Highest Arrest Rate by Driver Age Group
if visualization_option == "👨‍🦳 Highest Arrest Rate by Driver Age Group":
    if not data.empty and 'age_group' in data.columns and 'stop_outcome' in data.columns:
        data['is_arrested'] = data['stop_outcome'].str.contains("arrest", case=False, na=False).astype(int)
        arrest_rate_by_age = (
            data.groupby('age_group')['is_arrested']
            .mean()
            .reset_index(name='Arrest Rate')
        )
        highest = arrest_rate_by_age.sort_values('Arrest Rate', ascending=False).iloc[0]
        # Fixed indexing using .loc
        st.write(
            f"Age group with highest arrest rate: **{highest.loc['age_group']}** (Rate: {highest.loc['Arrest Rate']:.2%})"
        )
        fig = px.bar(arrest_rate_by_age, x='age_group', y='Arrest Rate', title="Arrest Rate by Age Group")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for age group arrest analysis.")

# Gender Distribution of Drivers Stopped in Each Country
elif visualization_option == "What is the gender distribution of drivers stopped in each country?":
    if not data.empty and 'country_name' in data.columns and 'driver_gender' in data.columns:
        # Group by country and gender, count the stops
        gender_country = (
            data.groupby(['country_name', 'driver_gender'])
            .size()
            .reset_index(name='Count')
        )
        # Plot grouped bar chart
        fig = px.bar(
            gender_country,
            x="country_name",
            y="Count",
            color="driver_gender",
            barmode="group",
            title="Gender Distribution of Drivers Stopped in Each Country"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data for country or driver gender is missing.")
elif visualization_option == "Which race and gender combination has the highest search rate?":
    if not data.empty and all(col in data.columns for col in ['driver_race', 'driver_gender', 'search_conducted']):
        # Group by race and gender, take average of search_conducted as search rate
        search_rate = (
            data.groupby(['driver_race', 'driver_gender'])['search_conducted']
            .mean()
            .reset_index(name='Search Rate')
        )

        # Optional: just plot the chart without highlighting the highest combo
        fig = px.bar(
            search_rate,
            x='driver_race',
            y='Search Rate',
            color='driver_gender',
            barmode='group',
            title="Search Rate by Race and Gender"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Required data for race, driver gender, or search conducted is missing.")


st.header("🕒 Time & Duration Based Visualizations:")

time_duration_option = st.selectbox(
    "Choose visualization",
    [
        "What time of day sees the most traffic stops?",
        "What is the average stop duration for different violations?",
        "Are stops during the night more likely to lead to arrests?"
    ]
)

if time_duration_option == "What time of day sees the most traffic stops?":
    if not data.empty and 'timestamp' in data.columns: 
        data['time'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data['hour'] = data['time'].dt.hour
       
        # Count stops by counting rows per hour
        stops_per_hour = data.groupby('hour')['stop_time'].size().reset_index(name='Number of Stops')
        
        if not stops_per_hour.empty:
            fig = px.bar(
                stops_per_hour,
                x='hour',
                y='Number of Stops',
                labels={'hour': 'Hour of Day (24hr)', 'Number of Stops': 'Stops Count'},
                title="Traffic Stops by Hour of Day"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Missing or invalid 'stop_time' data.")
    else:
        st.warning("Missing 'timestamp' column or data is empty.")



elif time_duration_option == "What is the average stop duration for different violations?":
    if not data.empty and 'violation' in data.columns and 'stop_duration' in data.columns:
        
        # Map categorical durations to numeric (midpoints or assumed averages)
        duration_map = {
            "0-15 Min": 7.5,
            "16-30 Min": 23,
            "30+ Min": 40
        }
        data['stop_duration_num'] = data['stop_duration'].map(duration_map)
        
        # Compute average stop duration per violation
        avg_stop_duration = (
            data.groupby('violation')['stop_duration_num']
            .mean()
            .reset_index()
            .sort_values(by='stop_duration_num', ascending=False)
        )
        
        st.write(avg_stop_duration)
        
        # Visualization: bar chart of average stop durations
        fig = px.bar(
            avg_stop_duration,
            x='violation',
            y='stop_duration_num',
            labels={'violation': 'Violation Type', 'stop_duration_num': 'Average Stop Duration (minutes)'},
            title='Average Stop Duration by Violation'
        )
        
        # Improve chart readability for categorical data
        fig.update_xaxes(tickangle=45, categoryorder='total descending')
        fig.update_layout(
            xaxis_title="Violation Type",
            yaxis_title="Average Stop Duration (minutes)",
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Data is missing 'violation' or 'stop_duration' columns or is empty.")



elif time_duration_option == "Are stops during the night more likely to lead to arrests?":
    if not data.empty and 'timestamp' in data.columns and 'stop_outcome' in data.columns:
        # Extract hour from timestamp
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        
        # Define night (6 PM to 6 AM)
        data['time_period'] = data['hour'].apply(lambda x: 'Night' if (x >= 18 or x < 6) else 'Day')
        
        # Calculate arrest rates
        total_stops = data.groupby('time_period').size()
        arrests = data[data['stop_outcome'].str.contains("arrest", case=False, na=False)].groupby('time_period').size()
        arrest_rate = arrests / total_stops
        
        # Display results
        st.write("### Night vs Day Arrest Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Day Arrest Rate", f"{arrest_rate.get('Day', 0):.2%}")
        with col2:
            st.metric("Night Arrest Rate", f"{arrest_rate.get('Night', 0):.2%}")
        
        # Conclusion
        if arrest_rate.get('Night', 0) > arrest_rate.get('Day', 0):
            st.success("**Yes, Night stops are more likely to lead to arrests**")
        else:
            st.info("**No, Day stops are more likely to lead to arrests**")
        
        # Create visualization directly without separate DataFrame
        fig = px.bar(
            x=arrest_rate.index,
            y=arrest_rate.values,
            labels={'x': 'Time Period', 'y': 'Arrest Rate'},
            title="Arrest Rate: Night vs Day"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Missing required data: timestamp and stop_outcome columns.")


st.header("⚖️ Violation-Based:")

violation_option = st.selectbox(
    "Choose visualization",
    [
        "Which violations are most associated with searches or arrests?",
        "Which violations are most common among younger drivers (<25)?",
        "Is there a violation that rarely results in search or arrest?"
    ]
)

if violation_option == "Which violations are most associated with searches or arrests?":
    if not data.empty and 'search_conducted' in data.columns and 'violation' in data.columns:
        
        # Convert and calculate
        data['search_conducted'] = data['search_conducted'] == 1
        
        summary = data.groupby('violation').agg(
            total_stops=('violation', 'size'),
            num_searches=('search_conducted', 'sum'),
            num_arrests=('stop_outcome', lambda x: (x == 'Arrest').sum())
        ).reset_index()
        
        summary['search_rate'] = summary['num_searches'] / summary['total_stops']
        summary['arrest_rate'] = summary['num_arrests'] / summary['total_stops']
        
        # Find top violations
        max_search_rate = summary['search_rate'].max()
        max_arrest_rate = summary['arrest_rate'].max()
        
        top_search_rows = summary[summary['search_rate'] == max_search_rate]
        top_arrest_rows = summary[summary['arrest_rate'] == max_arrest_rate]
        
        top_search = top_search_rows['violation'].values[0]
        top_arrest = top_arrest_rows['violation'].values[0]
        
        # Single grouped bar chart
        fig = px.bar(
            summary, 
            x='violation', 
            y=['search_rate', 'arrest_rate'],
            barmode='group',
            title='Search and Arrest Rates by Violation'
        )
        
        # Display
        st.write(f"**Most searched violation:** {top_search}")
        st.write(f"**Most arrested violation:** {top_arrest}")
        st.dataframe(summary[['violation', 'search_rate', 'arrest_rate']])
        st.plotly_chart(fig, use_container_width=True, key="violation_chart")
    
    else:
        st.warning("Missing data columns")


elif violation_option == "Which violations are most common among younger drivers (<25)?":
    if not data.empty and 'violation' in data.columns and 'driver_age' in data.columns:
        # Filter for younger drivers (<25)
        young_drivers = data[data['driver_age'] < 25]

        # Calculate count of each violation
        violations_count = young_drivers['violation'].value_counts()

        # Display results
        st.title('Most Common Violations Among Younger Drivers (<25)')

        st.write('### Violation Counts')
        st.dataframe(violations_count.rename_axis('violation').reset_index(name='count'))

        # Bar chart
        st.write('### Bar Chart of Violations')
        st.bar_chart(violations_count)

        if not violations_count.empty:
            most_common_violation = violations_count.index[0]
            st.write("### Analysis")
            st.write(f"The most common violation among younger drivers (<25) is **{most_common_violation}**.")
        else:
            st.write("No violations found for younger drivers.")
    else:
        st.warning("Missing data columns")


elif violation_option == "Is there a violation that rarely results in search or arrest?":
    if not data.empty and all(col in data.columns for col in ['violation', 'search_conducted', 'stop_outcome']):
        data['search_conducted'] = data['search_conducted'].astype(bool)
        data['arrested'] = data['stop_outcome'].str.contains("arrest", case=False, na=False)
        
        summary = data.groupby('violation').agg(
            search_rate=('search_conducted', 'mean'),
            arrest_rate=('arrested', 'mean')
        ).reset_index()
        
        # Sort ascending and take lowest 5 for search and arrests
        rare_search = summary.sort_values('search_rate').head(5)
        rare_arrest = summary.sort_values('arrest_rate').head(5)
        
        st.write("Violations Rarely Resulting in Search or Arrest (Lowest Rates)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("Top 5 Violations with Lowest Search Rates")
            st.dataframe(rare_search[['violation', 'search_rate']])
        
        with col2:
            st.write("Top 5 Violations with Lowest Arrest Rates")
            st.dataframe(rare_arrest[['violation', 'arrest_rate']])
    else:
        st.warning("Data is missing required columns.")

st.header("🌍 Location-Based")

location_based = st.selectbox(
    "Choose visualization",
    [
        "Which countries report the highest rate of drug-related stops?",
        "What is the arrest rate by country and violation?",
        "Which country has the most stops with search conducted?"
    ]
)

if location_based == "Which countries report the highest rate of drug-related stops?":
    if not data.empty and 'country_name' in data.columns and 'drugs_related_stop' in data.columns:

        # Calculate drug-related rate by country
        summary = data.groupby('country_name')['drugs_related_stop'].mean().reset_index(name='drug_rate')
        
        # Sort by drug_rate descending
        summary = summary.sort_values('drug_rate', ascending=False)

        # Display results
        st.title('Countries with Highest Rate of Drug-Related Stops')

        st.write('### Drug-Related Stop Rates by Country')
        st.dataframe(summary)

        # Bar chart
        st.write('### Bar Chart of Drug Rates')
        st.bar_chart(summary.set_index('country_name')['drug_rate'])

        # Highlight the highest
        highest_country = summary.iloc[0]['country_name']
        highest_rate = summary.iloc[0]['drug_rate']
        st.write(f"The country with the highest rate is **{highest_country}** with a rate of {highest_rate:.2%}.")


elif location_based == "What is the arrest rate by country and violation?":
    if not data.empty and 'country_name' in data.columns and 'violation' in data.columns:

        # Calculate total stops by country and violation
        total_stops = data.groupby(['country_name', 'violation']).size().reset_index(name='total_stops')

        # Calculate number of arrests by country and violation
        arrests = data[data['stop_outcome'] == 'Arrest'].groupby(['country_name', 'violation']).size().reset_index(name='arrests')

        # Merge and calculate arrest rate
        summary = total_stops.merge(arrests, on=['country_name', 'violation'], how='left').fillna(0)
        summary['arrest_rate'] = summary['arrests'] / summary['total_stops']

        # Streamlit display
        st.title('Arrest Rate by Country and Violation')

        st.write('### Summary Table')
        st.dataframe(summary)

        # Create grouped bar chart
        fig = px.bar(
            summary,
            x='violation',
            y='arrest_rate',
            color='country_name',
            barmode='group',
            title='Arrest Rates by Violation and Country',
            labels={
                'arrest_rate': 'Arrest Rate',
                'violation': 'Violation',
                'country_name': 'Country'
            },
            height=500
        )

        st.write('### Bar Chart of Arrest Rates')
        st.plotly_chart(fig)

elif location_based == "Which country has the most stops with search conducted?":
    if not data.empty and 'country_name' in data.columns and 'search_conducted' in data.columns:

        # Filter rows where search was conducted
        search_stops = data[data["search_conducted"] == True]

        # Count searches per country
        search_counts = search_stops["country_name"].value_counts().reset_index()
        search_counts.columns = ["Country", "Search_Count"]
        top_country = search_counts["Country"].values[0]
        st.write(f"Most stops with search conducted is {top_country}")


        # Bar chart
        fig = px.bar(
            search_counts,
            x="Country",
            y="Search_Count",
            title="Searches Conducted by Country",
            labels={"Search_Count": "Number of Searches"},
            color="Country"
        )

        st.plotly_chart(fig, use_container_width=True)

st.header("Traffic Stop and Driver Violation Analysis Overview")

stop_violation = st.selectbox(
    "Choose visualization",
    [
        "Yearly Breakdown of Stops and Arrests by Country",
        "Driver Violation Trends Based on Age and Race",
        "Time Period Analysis of Stops, Number of Stops by Year,Month, Hour of the Day",
        "Violations with High Search and Arrest Rates",
        "Driver Demographics by Country (Age, Gender, and Race)",
        "Top 5 Violations with Highest Arrest Rates"
    ]
)
if stop_violation == "Yearly Breakdown of Stops and Arrests by Country":
    if not data.empty and 'country_name' in data.columns and 'stop_date' in data.columns:

        query1 = '''
        WITH yearly_stats AS (
    SELECT 
        country_name,
        YEAR(stop_date) AS year,
        COUNT(DISTINCT CONCAT(country_name, '_', stop_date, '_', stop_time, '_', vehicle_number)) AS total_stops,
        COUNT(DISTINCT CASE WHEN LOWER(stop_outcome) LIKE '%%arrest%%'
            THEN CONCAT(country_name, '_', stop_date, '_', stop_time, '_', vehicle_number) END) AS total_arrests
    FROM project.traffic_stop_log
    GROUP BY country_name, YEAR(stop_date)
),
ranked_stats AS (
    SELECT 
        country_name,
        year,
        total_stops,
        total_arrests,
        RANK() OVER (PARTITION BY year ORDER BY total_stops DESC) AS stop_rank,
        RANK() OVER (PARTITION BY year ORDER BY total_arrests DESC) AS arrest_rank
    FROM yearly_stats
)
SELECT * FROM ranked_stats;


        '''

        st.title("Yearly Breakdown of Stops and Arrests by Country")
        st.markdown(
            """
            This report presents a yearly breakdown of traffic stops and arrests by country (by using window function).
            """
        )

        # Load data
        data1 = fetch_data(query1)

        if not data1.empty:
            # Show full table
            st.dataframe(data1)

        else:
            st.warning("No data available or connection error.")

elif stop_violation == "Driver Violation Trends Based on Age and Race":
    if not data.empty and 'driver_age' in data.columns and 'driver_race' in data.columns and 'violation' in data.columns:

        query2 = '''
        WITH violation_summary AS (
            SELECT 
                driver_age,
                driver_race,
                violation,
                COUNT(*) AS total_violations
            FROM project.traffic_stop_log
            GROUP BY driver_age, driver_race, violation
        )
        SELECT *
        FROM violation_summary
        ORDER BY driver_age, driver_race, violation;
        '''

        st.title("Driver Violation Trends by Age and Race")

        data2 = fetch_data(query2)  # use a new variable to avoid overwriting `data`

        if not data2.empty:
            races = data2['driver_race'].unique()
            selected_race = st.selectbox('Select Driver Race', options=sorted(races))

            filtered_data = data2[data2['driver_race'] == selected_race]

            fig = px.bar(
                filtered_data,
                x='driver_age',
                y='total_violations',
                color='violation',
                barmode='group',
                title=f'Violation Trends by Age for Race: {selected_race}'
            )
            st.plotly_chart(fig)
        else:
            st.warning("No data available.")

elif stop_violation == "Time Period Analysis of Stops, Number of Stops by Year,Month, Hour of the Day":
    if not data.empty and 'stop_date' in data.columns and 'stop_time' in data.columns:

        # Query for stops by year
        year_query = """
        SELECT YEAR(stop_date) AS year, COUNT(DISTINCT stop_date) AS stops
        FROM project.traffic_stop_log
        GROUP BY YEAR(stop_date)
        ORDER BY year
        """
        df_year = fetch_data(year_query)

        # Query for stops by month
        month_query = """
        SELECT MONTH(stop_date) AS month, COUNT(DISTINCT stop_date) AS stops
        FROM project.traffic_stop_log
        GROUP BY MONTH(stop_date)
        ORDER BY month
        """
        df_month = fetch_data(month_query)

        # Query for stops by hour
        hour_query = """
        SELECT HOUR(stop_time) AS hour, COUNT(DISTINCT CONCAT(stop_date, ' ', stop_time)) AS stops
        FROM project.traffic_stop_log
        GROUP BY HOUR(stop_time)
        ORDER BY hour
        """
        df_hour = fetch_data(hour_query)


        # Display results
        if not df_year.empty:
            st.write('### Stops by Year')
            st.dataframe(df_year)
            fig_year = px.bar(df_year, x='year', y='stops', title='Number of Stops by Year',
                              labels={'year': 'Year', 'stops': 'Number of Stops'})
            st.plotly_chart(fig_year)

        if not df_month.empty:
            st.write('### Stops by Month')
            st.dataframe(df_month)
            fig_month = px.bar(df_month, x='month', y='stops', title='Number of Stops by Month',
                               labels={'month': 'Month', 'stops': 'Number of Stops'})
            st.plotly_chart(fig_month)

        if not df_hour.empty:
            st.write('### Stops by Hour of the Day')
            st.dataframe(df_hour)
            fig_hour = px.bar(df_hour, x='hour', y='stops', title='Number of Stops by Hour',
                              labels={'hour': 'Hour of Day', 'stops': 'Number of Stops'})
            st.plotly_chart(fig_hour)

elif stop_violation == "Violations with High Search and Arrest Rates":
    if not data.empty and 'violation' in data.columns and 'search_conducted' in data.columns:

        query3 = """
        SELECT
        violation,
        search_rate,
        arrest_rate,
        RANK() OVER (ORDER BY search_rate DESC) AS search_rank,
        RANK() OVER (ORDER BY arrest_rate DESC) AS arrest_rank
        FROM (
            SELECT
            violation,
            AVG(CASE WHEN search_conducted = 'TRUE' THEN 1.0 ELSE 0.0 END) AS search_rate,
            AVG(CASE WHEN stop_outcome = 'Arrest' THEN 1.0 ELSE 0.0 END) AS arrest_rate
            FROM project.traffic_stop_log
            GROUP BY violation
        ) sub
        ORDER BY search_rate DESC, arrest_rate DESC
        """

        # Execute query
        result = fetch_data(query3)

        # Display results
        st.title('Violations with High Search and Arrest Rates (Using Window Functions)')

        st.write('### Rates and Ranks by Violation')
        st.dataframe(result)

        # Highlight high rates (e.g., top 3 ranks)
        high_search = result[result['search_rank'] <= 3].sort_values(by='search_rank')
        high_arrest = result[result['arrest_rank'] <= 3].sort_values(by='arrest_rank')


        st.write('### Violations with Highest Search Rates (Top 3)')
        st.dataframe(high_search[['violation', 'search_rate', 'search_rank']])

        st.write('### Violations with Highest Arrest Rates (Top 3)')
        st.dataframe(high_arrest[['violation', 'arrest_rate', 'arrest_rank']])

        # Bar chart for search and arrest rates
        chart_data = result.melt(id_vars='violation', value_vars=['search_rate', 'arrest_rate'],
                                var_name='Rate Type', value_name='Rate')

        fig = px.bar(chart_data,
                     x='violation',
                     y='Rate',
                     color='Rate Type',
                     barmode='group',
                     title='Search and Arrest Rates by Violation')

        st.write('### Visual Comparison of Rates')
        st.plotly_chart(fig)
    else:
        st.write("Data is empty or required columns are missing.")

elif stop_violation == "Driver Demographics by Country (Age, Gender, and Race)":
    required = {"driver_age", "driver_gender", "driver_race", "country_name"}
    if not data.empty and required.issubset(data.columns):
        # numeric age
        data["driver_age"] = pd.to_numeric(data["driver_age"], errors="coerce")

        # establish complete category lists (seen in data) for stable columns
        genders = sorted(data["driver_gender"].unique().tolist())
        races = sorted(data["driver_race"].unique().tolist())

        # Gender % by country (0–100), aligned columns
        gender_pct = (
            data.groupby("country_name")["driver_gender"]
                .value_counts(normalize=True)
                .mul(100)
                .rename("pct")
                .reset_index()
                .pivot(index="country_name", columns="driver_gender", values="pct")
                .reindex(columns=genders)              # consistent column order
                .fillna(0.0)
                .round(2)
                .reset_index()
        )

        # Race % by country (0–100), aligned columns
        race_pct = (
            data.groupby("country_name")["driver_race"]
                .value_counts(normalize=True)
                .mul(100)
                .rename("pct")
                .reset_index()
                .pivot(index="country_name", columns="driver_race", values="pct")
                .reindex(columns=races)
                .fillna(0.0)
                .round(2)
                .reset_index()
        )

        # Age stats rounded, index aligned
        age_stats = (
            data.groupby("country_name", as_index=False)["driver_age"]
                .agg(avg_age=("driver_age", "mean"), median_age=("driver_age", "median"))
                .round(1)
        )

        # Optional: sort countries alphabetically for all outputs
        gender_pct = gender_pct.sort_values("country_name")
        race_pct = race_pct.sort_values("country_name")
        age_stats = age_stats.sort_values("country_name")

        st.subheader("Gender Distribution (%)")
        st.dataframe(gender_pct)
        st.bar_chart(gender_pct.set_index("country_name"))  # columns now consistent

        st.subheader("Race Distribution (%)")
        st.dataframe(race_pct)
        st.bar_chart(race_pct.set_index("country_name"))

        st.subheader("Driver Age Statistics")
        st.dataframe(age_stats)
        st.bar_chart(age_stats.set_index("country_name")[["avg_age"]])
    else:
        st.warning("Required columns missing or data is empty")

elif stop_violation == "Top 5 Violations with Highest Arrest Rates":
    if not data.empty and 'violation' in data.columns:
        arrest_rates = data.groupby('violation')['stop_outcome'].apply(lambda x: (x == 'Arrest').mean() * 100).round(2).reset_index(name='arrest_rate')

        # Get top 5
        top_5 = arrest_rates.sort_values('arrest_rate', ascending=False).head(5)

        # Display results
        st.title('Top 5 Violations with Highest Arrest Rates')

        st.write('### Arrest Rates (%)')
        st.dataframe(top_5)

        # Bar chart
        fig = px.bar(top_5, x='violation', y='arrest_rate',
                    title='Top 5 Violations by Arrest Rate',
                    labels={'arrest_rate': 'Arrest Rate (%)', 'violation': 'Violation'})

        st.plotly_chart(fig)


st.header("📝 Add New Police Log & Predict Outcome and Violation")  

with st.form("new_log_form"):  
    stop_date = st.date_input("Stop Date")  
    stop_time = st.time_input("Stop Time")  
    county_name = st.selectbox("County Name",["Canada","India","USA"])  
    driver_gender = st.selectbox("Driver Gender", ["male", "female"])  
    driver_age = st.number_input("Driver Age", min_value=16, max_value=80, value=27)  
    driver_race = st.selectbox("Driver Race", ["Asian","Other","Black","White","Hispanic",])  
    search_conducted = st.selectbox("Was a Search Conducted?", ["0", "1"])  
    search_type = st.selectbox("Search Type",["Vehicle Search","Frisk","No Search"])  
    drugs_related_stop = st.selectbox("Was it Drug Related?", ["0", "1"])  
    stop_duration = st.selectbox("Stop Duration", data['stop_duration'].unique())  
    vehicle_number = st.text_input("Vehicle Number")  
    timestamp = pd.Timestamp.now()  

    submitted = st.form_submit_button("Predict Stop Outcome & Violation")  

if submitted:  
    filtered_data = data[
        (data['driver_gender'] == driver_gender) &
        (data['driver_age'] == driver_age) &
        (data['search_conducted'] == int(search_conducted)) &
        (data['stop_duration'] == stop_duration) &
        (data['drugs_related_stop'] == int(drugs_related_stop))
    ] 

    if not filtered_data.empty:
        predicted_outcome = filtered_data['stop_outcome'].mode()
        predicted_violation = filtered_data['violation'].mode()
    else:
        predicted_outcome = "Warning"
        predicted_violation = "Speeding"

    search_text = "A search was conducted" if int(search_conducted) else "No search was conducted"
    drug_text = "was drug-related" if int(drugs_related_stop) else "was not drug-related"

    st.markdown(f"""
### Prediction Summary

- Predicted Violation: {predicted_violation}
- Predicted Stop Outcome: {predicted_outcome}

🚓 A {driver_age}-year-old {driver_gender} driver in {county_name} was stopped at {stop_time.strftime("%I:%M %p")} on {stop_date}.
{search_text}, and the stop {drug_text}.
Stop duration: **{stop_duration}**
Vehicle Number: **{vehicle_number}**
""")



