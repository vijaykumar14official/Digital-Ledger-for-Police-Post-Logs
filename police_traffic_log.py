# ==============================================================
#  🔐 Securecheck: Police Check Post Digital Ledger
# A Streamlit-based dashboard connected to MySQL database
# For analyzing traffic stops, violations, arrests, and predictions
# ==============================================================

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.express as px

# -----------------------------
# 📌 Database Connection
# -----------------------------
def create_connection():
    """
    Create and return a SQLAlchemy connection to the MySQL database.
    Update the connection string to match your DB credentials if required.
    """
    try:
        # NOTE: This uses the mysqlclient driver (mysql+mysqldb). If you prefer pymysql use:
        # create_engine("mysql+pymysql://root:password@localhost:3306/project")
        engine = create_engine("mysql+mysqldb://root:admin@localhost:3306/project")
        connection = engine.connect()
        return connection
    except Exception as e:
        # Show error in Streamlit UI but don't crash the app
        st.error(f"Database Connection Error: {e}")
        return None

# -----------------------------
# 📌 Fetch Data from Database
# -----------------------------
def fetch_data(query):
    """
    Execute query and return results as a pandas DataFrame.
    On failure, returns an empty DataFrame.
    """
    connection = create_connection()
    if connection:
        try:
            # Read SQL into DataFrame using SQLAlchemy connection
            df = pd.read_sql(query, connection)
            return df
        finally:
            # Always close the connection
            connection.close()
    else:
        # Return empty DataFrame if connection failed
        return pd.DataFrame()

# -----------------------------
# 📌 Streamlit Page Setup
# -----------------------------
st.set_page_config(page_title="Securecheck Police Dashboard", layout="wide")
st.title(" 🛡️ Securecheck: Police Check Post Digital Ledger")
st.markdown("Real-time monitoring and insights for law enforcement 🚔")

# -----------------------------
# 🗂️ Police Logs Overview (Load & display full table)
# -----------------------------
st.header("🗂️ Police Logs Overview")
query = "SELECT * FROM traffic_stop_log"
data = fetch_data(query)  # Load data from DB

# Safe conversion / extraction of stop_time from timestamp column if present
if not data.empty and 'timestamp' in data.columns:
    # Ensure timestamp is datetime, coerce errors to NaT
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data['stop_time'] = data['timestamp'].dt.time
else:
    # If timestamp absent, ensure stop_time exists as column to avoid downstream errors
    if 'stop_time' not in data.columns:
        data['stop_time'] = pd.NA

# Show the raw table for inspection
st.dataframe(data, use_container_width=True)

# -----------------------------
# 📊 Dashboard Metrics (KPIs)
# -----------------------------
st.header("📊 Dashboard Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    # Total number of stops (rows in the dataframe)
    total_stops = data.shape[0]
    st.metric("Total Police Stops", total_stops)

with col2:
    # Count rows where stop_outcome contains "arrest" (case-insensitive)
    if 'stop_outcome' in data.columns:
        arrests = data['stop_outcome'].str.contains("arrest", case=False, na=False).sum()
    else:
        arrests = 0
    st.metric("Total Arrests", arrests)

with col3:
    # Count rows where stop_outcome contains "warning"
    if 'stop_outcome' in data.columns:
        warnings = data['stop_outcome'].str.contains("warning", case=False, na=False).sum()
    else:
        warnings = 0
    st.metric("Total Warnings", warnings)

with col4:
    # Count drug-related stops (assumes 1 indicates drug-related)
    if 'drugs_related_stop' in data.columns:
        drug_related_stop = (data['drugs_related_stop'] == 1).sum()
    else:
        drug_related_stop = 0
    st.metric("Total Drug related stop", drug_related_stop)

# -----------------------------
# 🚗 Vehicle Based Report
# -----------------------------
st.header("🗂️ Vehicle based report")
vehicle_based_report = st.selectbox(
    "Choose Visualization:",
    [
        "What are the top 10 vehicle_Number involved in drug-related stops?",
        "Which vehicles were most frequently searched?"
    ]
)

# Option 1: Top 10 vehicle numbers involved in drug-related stops
if vehicle_based_report == "What are the top 10 vehicle_Number involved in drug-related stops?":
    if not data.empty and 'vehicle_number' in data.columns and 'drugs_related_stop' in data.columns:
        # Filter rows where drugs_related_stop == 1
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

# Option 2: Which vehicles were most frequently searched?
elif vehicle_based_report == "Which vehicles were most frequently searched?":
    if not data.empty and 'vehicle_number' in data.columns and 'search_conducted' in data.columns:
        # Filter rows where a search was conducted (assuming 1 == searched)
        searched_vehicles = data[data['search_conducted'] == 1]
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

# -----------------------------
# 📊 Visual Insights
# -----------------------------
st.header("🗂️ Visual Insights")

visualization_option = st.selectbox(
    "Choose a visualization:",
    [
        "👨‍🦳 Highest Arrest Rate by Driver Age Group",
        "What is the gender distribution of drivers stopped in each country?",
        "Which race and gender combination has the highest search rate?"
    ]
)

# Arrest Rate by Age Group
if visualization_option == "👨‍🦳 Highest Arrest Rate by Driver Age Group":
    if not data.empty and 'age_group' in data.columns and 'stop_outcome' in data.columns:
        # Create binary column for arrested yes/no
        data['is_arrested'] = data['stop_outcome'].str.contains("arrest", case=False, na=False).astype(int)
        arrest_rate_by_age = (
            data.groupby('age_group')['is_arrested']
            .mean()
            .reset_index(name='Arrest Rate')
        )
        # Safely select highest arrest rate row
        if not arrest_rate_by_age.empty:
            highest = arrest_rate_by_age.sort_values('Arrest Rate', ascending=False).iloc[0]
            st.write(
                f"Age group with highest arrest rate: **{highest.loc['age_group']}** (Rate: {highest.loc['Arrest Rate']:.2%})"
            )
        fig = px.bar(arrest_rate_by_age, x='age_group', y='Arrest Rate', title="Arrest Rate by Age Group")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Insufficient data for age group arrest analysis.")

# Gender distribution by country
elif visualization_option == "What is the gender distribution of drivers stopped in each country?":
    if not data.empty and 'country_name' in data.columns and 'driver_gender' in data.columns:
        gender_country = (
            data.groupby(['country_name', 'driver_gender'])
            .size()
            .reset_index(name='Count')
        )
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

# Race & gender combination search rate
elif visualization_option == "Which race and gender combination has the highest search rate?":
    if not data.empty and all(col in data.columns for col in ['driver_race', 'driver_gender', 'search_conducted']):
        # Average of search_conducted (1/0) gives search rate
        search_rate = (
            data.groupby(['driver_race', 'driver_gender'])['search_conducted']
            .mean()
            .reset_index(name='Search Rate')
        )
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

# -----------------------------
# 🕒 Time & Duration Based Visualizations
# -----------------------------
st.header("🕒 Time & Duration Based Visualizations:")

time_duration_option = st.selectbox(
    "Choose visualization",
    [
        "What time of day sees the most traffic stops?",
        "What is the average stop duration for different violations?",
        "Are stops during the night more likely to lead to arrests?"
    ]
)

# What time of day sees the most traffic stops?
if time_duration_option == "What time of day sees the most traffic stops?":
    if not data.empty and 'timestamp' in data.columns:
        # Convert timestamp safely
        data['time'] = pd.to_datetime(data['timestamp'], errors='coerce')
        data['hour'] = data['time'].dt.hour

        # Count stops per hour using stop_time or hour
        if 'hour' in data.columns:
            stops_per_hour = data.groupby('hour').size().reset_index(name='Number of Stops')
        else:
            stops_per_hour = pd.DataFrame()

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

# Average stop duration for different violations
elif time_duration_option == "What is the average stop duration for different violations?":
    if not data.empty and 'violation' in data.columns and 'stop_duration' in data.columns:
        # Map categorical durations to numeric midpoints for averaging
        duration_map = {
            "0-15 Min": 7.5,
            "16-30 Min": 23,
            "30+ Min": 40
        }
        # Create numeric representation; values not in map become NaN
        data['stop_duration_num'] = data['stop_duration'].map(duration_map)

        # Group by violation and compute mean of numeric duration
        avg_stop_duration = (
            data.groupby('violation')['stop_duration_num']
            .mean()
            .reset_index()
            .sort_values(by='stop_duration_num', ascending=False)
        )

        st.write(avg_stop_duration)

        # Visualize average stop duration per violation
        fig = px.bar(
            avg_stop_duration,
            x='violation',
            y='stop_duration_num',
            labels={'violation': 'Violation Type', 'stop_duration_num': 'Average Stop Duration (minutes)'},
            title='Average Stop Duration by Violation'
        )
        # Improve readability
        fig.update_xaxes(tickangle=45, categoryorder='total descending')
        fig.update_layout(xaxis_title="Violation Type", yaxis_title="Average Stop Duration (minutes)", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Data is missing 'violation' or 'stop_duration' columns or is empty.")

# Are night stops more likely to lead to arrests?
elif time_duration_option == "Are stops during the night more likely to lead to arrests?":
    if not data.empty and 'timestamp' in data.columns and 'stop_outcome' in data.columns:
        data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
        # Define night as 18:00-05:59
        data['time_period'] = data['hour'].apply(lambda x: 'Night' if (x >= 18 or x < 6) else 'Day')

        # Calculate arrest rates per time period
        total_stops = data.groupby('time_period').size()
        arrests = data[data['stop_outcome'].str.contains("arrest", case=False, na=False)].groupby('time_period').size()
        arrest_rate = arrests / total_stops

        st.write("### Night vs Day Arrest Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Day Arrest Rate", f"{arrest_rate.get('Day', 0):.2%}")
        with col2:
            st.metric("Night Arrest Rate", f"{arrest_rate.get('Night', 0):.2%}")

        # Conclusion message
        if arrest_rate.get('Night', 0) > arrest_rate.get('Day', 0):
            st.success("**Yes, Night stops are more likely to lead to arrests**")
        else:
            st.info("**No, Day stops are more likely to lead to arrests**")

        # Plot comparison
        # Convert arrest_rate Series to DataFrame for plotting if necessary
        ar_df = arrest_rate.reset_index()
        ar_df.columns = ['Time Period', 'Arrest Rate']
        fig = px.bar(ar_df, x='Time Period', y='Arrest Rate', title="Arrest Rate: Night vs Day")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Missing required data: timestamp and stop_outcome columns.")

# -----------------------------
# ⚖️ Violation-Based Analysis
# -----------------------------
st.header("⚖️ Violation-Based:")

violation_option = st.selectbox(
    "Choose visualization",
    [
        "Which violations are most associated with searches or arrests?",
        "Which violations are most common among younger drivers (<25)?",
        "Is there a violation that rarely results in search or arrest?"
    ]
)

# Violations associated with searches or arrests
if violation_option == "Which violations are most associated with searches or arrests?":
    if not data.empty and 'search_conducted' in data.columns and 'violation' in data.columns:
        # Normalize search_conducted to boolean (True if 1)
        data['search_conducted'] = data['search_conducted'] == 1

        summary = data.groupby('violation').agg(
            total_stops=('violation', 'size'),
            num_searches=('search_conducted', 'sum'),
            num_arrests=('stop_outcome', lambda x: (x.str.contains("arrest", case=False, na=False)).sum() if x.dtype == object else (x == 'Arrest').sum())
        ).reset_index()

        summary['search_rate'] = summary['num_searches'] / summary['total_stops']
        summary['arrest_rate'] = summary['num_arrests'] / summary['total_stops']

        # Find top violations by rates
        if not summary.empty:
            max_search_rate = summary['search_rate'].max()
            max_arrest_rate = summary['arrest_rate'].max()

            top_search_rows = summary[summary['search_rate'] == max_search_rate]
            top_arrest_rows = summary[summary['arrest_rate'] == max_arrest_rate]

            top_search = top_search_rows['violation'].values[0] if not top_search_rows.empty else None
            top_arrest = top_arrest_rows['violation'].values[0] if not top_arrest_rows.empty else None
        else:
            top_search = None
            top_arrest = None

        fig = px.bar(summary, x='violation', y=['search_rate', 'arrest_rate'], barmode='group', title='Search and Arrest Rates by Violation')

        st.write(f"**Most searched violation:** {top_search}")
        st.write(f"**Most arrested violation:** {top_arrest}")
        st.dataframe(summary[['violation', 'search_rate', 'arrest_rate']])
        st.plotly_chart(fig, use_container_width=True, key="violation_chart")
    else:
        st.warning("Missing data columns")

# Violations among younger drivers (<25)
elif violation_option == "Which violations are most common among younger drivers (<25)?":
    if not data.empty and 'violation' in data.columns and 'driver_age' in data.columns:
        young_drivers = data[data['driver_age'] < 25]
        violations_count = young_drivers['violation'].value_counts()

        st.title('Most Common Violations Among Younger Drivers (<25)')
        st.write('### Violation Counts')
        st.dataframe(violations_count.rename_axis('violation').reset_index(name='count'))
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

# Violations that rarely result in search or arrest
elif violation_option == "Is there a violation that rarely results in search or arrest?":
    if not data.empty and all(col in data.columns for col in ['violation', 'search_conducted', 'stop_outcome']):
        data['search_conducted'] = data['search_conducted'].astype(bool)
        data['arrested'] = data['stop_outcome'].str.contains("arrest", case=False, na=False)

        summary = data.groupby('violation').agg(
            search_rate=('search_conducted', 'mean'),
            arrest_rate=('arrested', 'mean')
        ).reset_index()

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

# -----------------------------
# 🌍 Location-Based Analysis
# -----------------------------
st.header("🌍 Location-Based")

location_based = st.selectbox(
    "Choose visualization",
    [
        "Which countries report the highest rate of drug-related stops?",
        "What is the arrest rate by country and violation?",
        "Which country has the most stops with search conducted?"
    ]
)

# Countries with highest rate of drug-related stops
if location_based == "Which countries report the highest rate of drug-related stops?":
    if not data.empty and 'country_name' in data.columns and 'drugs_related_stop' in data.columns:
        # Calculate mean (proportion) of drugs_related_stop per country
        summary = data.groupby('country_name')['drugs_related_stop'].mean().reset_index(name='drug_rate')
        summary = summary.sort_values('drug_rate', ascending=False)

        st.title('Countries with Highest Rate of Drug-Related Stops')
        st.write('### Drug-Related Stop Rates by Country')
        st.dataframe(summary)
        st.write('### Bar Chart of Drug Rates')
        st.bar_chart(summary.set_index('country_name')['drug_rate'])

        # Highlight the highest
        if not summary.empty:
            highest_country = summary.iloc[0]['country_name']
            highest_rate = summary.iloc[0]['drug_rate']
            st.write(f"The country with the highest rate is **{highest_country}** with a rate of {highest_rate:.2%}.")
    else:
        st.warning("Missing country_name or drugs_related_stop column.")

# Arrest rate by country and violation
elif location_based == "What is the arrest rate by country and violation?":
    if not data.empty and 'country_name' in data.columns and 'violation' in data.columns:
        total_stops = data.groupby(['country_name', 'violation']).size().reset_index(name='total_stops')
        arrests = data[data['stop_outcome'].str.contains("arrest", case=False, na=False)].groupby(['country_name', 'violation']).size().reset_index(name='arrests')
        summary = total_stops.merge(arrests, on=['country_name', 'violation'], how='left').fillna(0)
        summary['arrest_rate'] = summary['arrests'] / summary['total_stops']

        st.title('Arrest Rate by Country and Violation')
        st.write('### Summary Table')
        st.dataframe(summary)

        fig = px.bar(
            summary,
            x='violation',
            y='arrest_rate',
            color='country_name',
            barmode='group',
            title='Arrest Rates by Violation and Country',
            labels={'arrest_rate': 'Arrest Rate', 'violation': 'Violation', 'country_name': 'Country'},
            height=500
        )
        st.write('### Bar Chart of Arrest Rates')
        st.plotly_chart(fig)
    else:
        st.warning("Missing data for country or violation.")

# Country with most stops where search conducted
elif location_based == "Which country has the most stops with search conducted?":
    if not data.empty and 'country_name' in data.columns and 'search_conducted' in data.columns:
        # Ensure search_conducted is boolean or 0/1
        search_stops = data[data["search_conducted"] == True]
        search_counts = search_stops["country_name"].value_counts().reset_index()
        search_counts.columns = ["Country", "Search_Count"]
        if not search_counts.empty:
            top_country = search_counts["Country"].values[0]
            st.write(f"Most stops with search conducted is {top_country}")

            fig = px.bar(
                search_counts,
                x="Country",
                y="Search_Count",
                title="Searches Conducted by Country",
                labels={"Search_Count": "Number of Searches"},
                color="Country"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("No searches recorded in the dataset.")
    else:
        st.warning("Missing country_name or search_conducted column.")

# -----------------------------
# 📈 Traffic Stop and Driver Violation Analysis Overview (Advanced reports)
# -----------------------------
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

# Yearly breakdown using SQL window functions (executed in DB)
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
        st.markdown("This report presents a yearly breakdown of traffic stops and arrests by country (using window functions).")
        data1 = fetch_data(query1)
        if not data1.empty:
            st.dataframe(data1)
        else:
            st.warning("No data available or connection error.")
    else:
        st.warning("Missing country_name or stop_date columns for yearly breakdown.")

# Driver Violation Trends Based on Age and Race
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
        data2 = fetch_data(query2)
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
            st.warning("No data available from query2.")
    else:
        st.warning("Missing driver_age, driver_race, or violation columns.")

# Time Period Analysis (Year, Month, Hour)
elif stop_violation == "Time Period Analysis of Stops, Number of Stops by Year,Month, Hour of the Day":
    if not data.empty and 'stop_date' in data.columns and 'stop_time' in data.columns:
        # Query stops by year
        year_query = """
        SELECT YEAR(stop_date) AS year, COUNT(DISTINCT stop_date) AS stops
        FROM project.traffic_stop_log
        GROUP BY YEAR(stop_date)
        ORDER BY year
        """
        df_year = fetch_data(year_query)

        # Query stops by month
        month_query = """
        SELECT MONTH(stop_date) AS month, COUNT(DISTINCT stop_date) AS stops
        FROM project.traffic_stop_log
        GROUP BY MONTH(stop_date)
        ORDER BY month
        """
        df_month = fetch_data(month_query)

        # Query stops by hour
        hour_query = """
        SELECT HOUR(stop_time) AS hour, COUNT(DISTINCT CONCAT(stop_date, ' ', stop_time)) AS stops
        FROM project.traffic_stop_log
        GROUP BY HOUR(stop_time)
        ORDER BY hour
        """
        df_hour = fetch_data(hour_query)

        # Display results & charts if available
        if not df_year.empty:
            st.write('### Stops by Year')
            st.dataframe(df_year)
            fig_year = px.bar(df_year, x='year', y='stops', title='Number of Stops by Year', labels={'year':'Year','stops':'Number of Stops'})
            st.plotly_chart(fig_year)
        if not df_month.empty:
            st.write('### Stops by Month')
            st.dataframe(df_month)
            fig_month = px.bar(df_month, x='month', y='stops', title='Number of Stops by Month', labels={'month':'Month','stops':'Number of Stops'})
            st.plotly_chart(fig_month)
        if not df_hour.empty:
            st.write('### Stops by Hour of the Day')
            st.dataframe(df_hour)
            fig_hour = px.bar(df_hour, x='hour', y='stops', title='Number of Stops by Hour', labels={'hour':'Hour of Day','stops':'Number of Stops'})
            st.plotly_chart(fig_hour)
    else:
        st.warning("Missing stop_date or stop_time columns for time period analysis.")

# Violations with High Search and Arrest Rates (SQL + window functions)
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
        result = fetch_data(query3)
        st.title('Violations with High Search and Arrest Rates (Using Window Functions)')
        st.write('### Rates and Ranks by Violation')
        st.dataframe(result)
        if not result.empty:
            high_search = result[result['search_rank'] <= 3].sort_values(by='search_rank')
            high_arrest = result[result['arrest_rank'] <= 3].sort_values(by='arrest_rank')
            st.write('### Violations with Highest Search Rates (Top 3)')
            st.dataframe(high_search[['violation', 'search_rate', 'search_rank']])
            st.write('### Violations with Highest Arrest Rates (Top 3)')
            st.dataframe(high_arrest[['violation', 'arrest_rate', 'arrest_rank']])

            chart_data = result.melt(id_vars='violation', value_vars=['search_rate', 'arrest_rate'],
                                    var_name='Rate Type', value_name='Rate')
            fig = px.bar(chart_data, x='violation', y='Rate', color='Rate Type', barmode='group', title='Search and Arrest Rates by Violation')
            st.write('### Visual Comparison of Rates')
            st.plotly_chart(fig)
        else:
            st.warning("No results returned from query3.")
    else:
        st.write("Data is empty or required columns are missing.")

# Driver Demographics by Country (Age, Gender, Race)
elif stop_violation == "Driver Demographics by Country (Age, Gender, and Race)":
    required = {"driver_age", "driver_gender", "driver_race", "country_name"}
    if not data.empty and required.issubset(data.columns):
        # Convert age to numeric
        data["driver_age"] = pd.to_numeric(data["driver_age"], errors="coerce")
        genders = sorted(data["driver_gender"].dropna().unique().tolist())
        races = sorted(data["driver_race"].dropna().unique().tolist())

        # Gender percentage by country (0-100)
        gender_pct = (
            data.groupby("country_name")["driver_gender"]
                .value_counts(normalize=True)
                .mul(100)
                .rename("pct")
                .reset_index()
                .pivot(index="country_name", columns="driver_gender", values="pct")
                .reindex(columns=genders)
                .fillna(0.0)
                .round(2)
                .reset_index()
        )

        # Race percentage by country
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

        # Age stats
        age_stats = (
            data.groupby("country_name", as_index=False)["driver_age"]
                .agg(avg_age=("driver_age", "mean"), median_age=("driver_age", "median"))
                .round(1)
        )

        # Display
        st.subheader("Gender Distribution (%)")
        st.dataframe(gender_pct)
        st.bar_chart(gender_pct.set_index("country_name"))

        st.subheader("Race Distribution (%)")
        st.dataframe(race_pct)
        st.bar_chart(race_pct.set_index("country_name"))

        st.subheader("Driver Age Statistics")
        st.dataframe(age_stats)
        st.bar_chart(age_stats.set_index("country_name")[["avg_age"]])
    else:
        st.warning("Required columns missing or data is empty")

# Top 5 Violations with Highest Arrest Rates
elif stop_violation == "Top 5 Violations with Highest Arrest Rates":
    if not data.empty and 'violation' in data.columns:
        arrest_rates = data.groupby('violation')['stop_outcome'].apply(lambda x: (x.str.contains("arrest", case=False, na=False)).mean() * 100 if x.dtype == object else (x == 'Arrest').mean() * 100).round(2).reset_index(name='arrest_rate')
        top_5 = arrest_rates.sort_values('arrest_rate', ascending=False).head(5)
        st.title('Top 5 Violations with Highest Arrest Rates')
        st.write('### Arrest Rates (%)')
        st.dataframe(top_5)
        fig = px.bar(top_5, x='violation', y='arrest_rate', title='Top 5 Violations by Arrest Rate', labels={'arrest_rate':'Arrest Rate (%)','violation':'Violation'})
        st.plotly_chart(fig)
    else:
        st.warning("Missing violation column.")

# -----------------------------
# 📝 Add New Police Log & Predict Outcome
# -----------------------------
st.header("📝 Add New Police Log & Predict Outcome and Violation")

with st.form("new_log_form"):
    # Collect input fields for a new stop
    stop_date = st.date_input("Stop Date")
    stop_time = st.time_input("Stop Time")
    county_name = st.selectbox("County Name", ["Canada", "India", "USA"])
    driver_gender = st.selectbox("Driver Gender", ["male", "female"])
    driver_age = st.number_input("Driver Age", min_value=16, max_value=100, value=27)
    driver_race = st.selectbox("Driver Race", ["Asian", "Other", "Black", "White", "Hispanic"])
    search_conducted = st.selectbox("Was a Search Conducted?", ["0", "1"])
    search_type = st.selectbox("Search Type", ["Vehicle Search", "Frisk", "No Search"])
    drugs_related_stop = st.selectbox("Was it Drug Related?", ["0", "1"])
    # For stop_duration, if data has options use them; otherwise provide defaults
    if not data.empty and 'stop_duration' in data.columns:
        stop_duration_options = data['stop_duration'].dropna().unique().tolist()
    else:
        stop_duration_options = ["0-15 Min", "16-30 Min", "30+ Min"]
    stop_duration = st.selectbox("Stop Duration", stop_duration_options)
    vehicle_number = st.text_input("Vehicle Number")
    timestamp = pd.Timestamp.now()

    submitted = st.form_submit_button("Predict Stop Outcome & Violation")

# Prediction logic (on submit)
if submitted:
    # Convert selectbox string inputs to int where necessary
    try:
        search_int = int(search_conducted)
    except:
        search_int = 0
    try:
        drugs_int = int(drugs_related_stop)
    except:
        drugs_int = 0

    # Filter historical data to find records similar to the new input
    # Note: filter uses exact matches - you can improve fuzzy matching or modelling later
    filtered_data = data[
        (data.get('driver_gender') == driver_gender) &
        (data.get('driver_age') == driver_age) &
        (data.get('search_conducted') == search_int) &
        (data.get('stop_duration') == stop_duration) &
        (data.get('drugs_related_stop') == drugs_int)
    ] if not data.empty else pd.DataFrame()

    # If we have similar past records, predict by mode (most frequent)
    if not filtered_data.empty:
        # .mode() returns Series; we pick the first mode if available
        try:
            predicted_outcome = filtered_data['stop_outcome'].mode().iloc[0]
        except Exception:
            predicted_outcome = filtered_data['stop_outcome'].mode().values if not filtered_data['stop_outcome'].mode().empty else "Warning"
        try:
            predicted_violation = filtered_data['violation'].mode().iloc[0]
        except Exception:
            predicted_violation = filtered_data['violation'].mode().values if not filtered_data['violation'].mode().empty else "Speeding"
    else:
        # Default guesses if no similar record found
        predicted_outcome = "Warning"
        predicted_violation = "Speeding"

    # whether search/drugs were present
    search_text = "A search was conducted" if search_int else "No search was conducted"
    drug_text = "was drug-related" if drugs_int else "was not drug-related"

    # Display the prediction summary
    st.markdown(f"""
### Prediction Summary

- Predicted Violation: **{predicted_violation}**
- Predicted Stop Outcome: **{predicted_outcome}**

🚓 A {driver_age}-year-old {driver_gender} driver in {county_name} was stopped at {stop_time.strftime("%I:%M %p")} on {stop_date}.
{search_text}, and the stop {drug_text}.
Stop duration: **{stop_duration}**
Vehicle Number: **{vehicle_number}**
""")


