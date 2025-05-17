import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from scipy import stats


class WeatherDataAnalyzer:
    def __init__(self, file_path=None):
        """Initialize the WeatherDataAnalyzer with optional file path."""
        self.data = None
        if file_path:
            self.load_data(file_path)

    def load_data(self, file_path):
        """Load weather data from CSV file."""
        try:
            # Read CSV file, parse dates
            self.data = pd.read_csv(file_path, parse_dates=['Date'])
            # Set Date as the index for time-based operations
            self.data.set_index('Date', inplace=True)

            # Convert percentage columns to proper decimal format if needed
            if 'Humidity' in self.data.columns and self.data['Humidity'].max() <= 1:
                print("Note: Humidity appears to be in decimal format (0-1)")
            elif 'Humidity' in self.data.columns:
                print("Note: Converting Humidity from percentage to decimal")
                self.data['Humidity'] = self.data['Humidity'] / 100

            print(
                f"Data loaded successfully with {self.data.shape[0]} records.")
            print(f"Columns: {', '.join(self.data.columns)}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False

    def clean_data(self):
        """Clean the data by removing missing values and outliers."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return False

        # Record initial shape
        initial_shape = self.data.shape

        # Remove rows with all NaN values
        self.data.dropna(how='all', inplace=True)

        # Fill missing values with column means for numeric data
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        self.data[numeric_cols] = self.data[numeric_cols].fillna(
            self.data[numeric_cols].mean())

        # Fill categorical columns with mode
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            self.data[col] = self.data[col].fillna(
                self.data[col].mode()[0] if not self.data[col].mode().empty else "Unknown")

        # Remove extreme outliers (Z-score > 3)
        for col in numeric_cols:
            if self.data[col].count() > 3:  # Need at least a few values to calculate z-score
                z_scores = np.abs(stats.zscore(self.data[col].dropna()))
                outlier_indices = self.data[col].dropna().index[z_scores > 3]
                if not outlier_indices.empty:
                    print(
                        f"Found {len(outlier_indices)} outliers in column {col}")
                    # Replace outliers with column median
                    self.data.loc[outlier_indices,
                                  col] = self.data[col].median()

        # Report cleaning results
        print(
            f"Data cleaning complete: {initial_shape[0] - self.data.shape[0]} rows removed")
        print(f"Final data shape: {self.data.shape}")
        return True

    def filter_by_date_range(self, start_date=None, end_date=None):
        """Filter data by date range."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None

        filtered_data = self.data.copy()

        if start_date:
            filtered_data = filtered_data[filtered_data.index >= pd.to_datetime(
                start_date)]

        if end_date:
            filtered_data = filtered_data[filtered_data.index <= pd.to_datetime(
                end_date)]

        print(
            f"Filtered data from {filtered_data.index.min()} to {filtered_data.index.max()}")
        print(f"Records: {filtered_data.shape[0]}")

        return filtered_data

    def resample_data(self, data=None, frequency='M'):
        """
        Resample data to specified frequency.
        frequency: 'W' for weekly, 'M' for monthly, 'Q' for quarterly, etc.
        """
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return None
            data = self.data

        # Create different aggregations based on column
        agg_dict = {
            'Temperature(°C)': 'mean',
            'Apparent Temperature(°C)': 'mean',
            'Humidity': 'mean',
            'Wind Speed(km/h)': 'mean',
            'Wind Bearing(degrees)': 'mean',
            'Visibility(km)': 'mean',
            'Pressure(millibars)': 'mean',
            'Loud Cover': 'mean'
        }

        # Only use columns that exist in the data
        agg_dict = {k: v for k, v in agg_dict.items() if k in data.columns}

        # For categorical columns, take the most common value
        categorical_cols = data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            agg_dict[col] = lambda x: x.mode(
            )[0] if not x.mode().empty else "Mixed"

        # Resample and aggregate
        resampled_data = data.resample(frequency).agg(agg_dict)

        freq_map = {'D': 'daily', 'W': 'weekly',
                    'M': 'monthly', 'Q': 'quarterly', 'Y': 'yearly'}
        print(
            f"Data resampled to {freq_map.get(frequency, frequency)} frequency")

        return resampled_data

    def detect_anomalies(self, column, threshold=2.0, data=None):
        """
        Detect anomalies in specified column using z-score method.
        threshold: Number of standard deviations to consider as anomaly
        """
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return None
            data = self.data

        if column not in data.columns:
            print(f"Column '{column}' not found in data.")
            return None

        # Calculate z-scores for the column
        z_scores = np.abs(stats.zscore(data[column].dropna()))

        # Find anomalies
        anomalies = data[column].dropna()[z_scores > threshold]

        print(
            f"Found {len(anomalies)} anomalies in {column} (threshold: {threshold} std)")

        return anomalies

    def calculate_rolling_average(self, column, window=7, data=None):
        """Calculate rolling average for specified column."""
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return None
            data = self.data

        if column not in data.columns:
            print(f"Column '{column}' not found in data.")
            return None

        rolling_avg = data[column].rolling(window=window, center=False).mean()
        print(f"Calculated {window}-day rolling average for {column}")

        return rolling_avg

    def calculate_feels_like_difference(self, data=None):
        """Calculate difference between actual and apparent (feels like) temperature."""
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return None
            data = self.data

        # Check if necessary columns exist
        if 'Temperature(°C)' in data.columns and 'Apparent Temperature(°C)' in data.columns:
            temp_diff = data['Temperature(°C)'] - \
                data['Apparent Temperature(°C)']
            print("Calculated difference between actual and apparent temperature")
            return temp_diff
        else:
            print("Temperature or Apparent Temperature columns not found")
            return None

    def plot_temperature_over_time(self, data=None, rolling_window=None, include_apparent=True):
        """Plot temperature over time with optional rolling average and apparent temperature."""
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return
            data = self.data

        if 'Temperature(°C)' not in data.columns:
            print("Temperature column not found in data.")
            return

        plt.figure(figsize=(12, 6))

        # Plot actual temperature
        plt.plot(data.index, data['Temperature(°C)'],
                 'b-', alpha=0.5, label='Actual Temperature')

        # Plot apparent temperature if requested and available
        if include_apparent and 'Apparent Temperature(°C)' in data.columns:
            plt.plot(data.index, data['Apparent Temperature(°C)'],
                     'g-', alpha=0.5, label='Apparent Temperature')

        # Plot rolling average if specified
        if rolling_window:
            rolling_avg = self.calculate_rolling_average(
                'Temperature(°C)', window=rolling_window, data=data)
            plt.plot(rolling_avg.index, rolling_avg, 'r-', linewidth=2,
                     label=f'{rolling_window}-day Rolling Average')

        plt.title('Temperature Over Time')
        plt.ylabel('Temperature (°C)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Format x-axis based on date range
        ax = plt.gca()
        date_range = (data.index.max() - data.index.min()).days

        if date_range <= 60:  # For data spanning up to 2 months
            ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            plt.xticks(rotation=45)
        elif date_range <= 365:  # For data spanning up to a year
            ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.xticks(rotation=45)
        else:  # For data spanning multiple years
            ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_weather_summary_distribution(self, data=None):
        """Plot distribution of weather summary categories."""
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return
            data = self.data

        if 'Summary' not in data.columns:
            print("Summary column not found in data.")
            return

        plt.figure(figsize=(12, 6))

        # Count occurrences of each weather summary
        summary_counts = data['Summary'].value_counts()

        # Create bar plot
        sns.barplot(x=summary_counts.index,
                    y=summary_counts.values, palette='viridis')

        plt.title('Distribution of Weather Conditions')
        plt.ylabel('Count')
        plt.xlabel('Weather Condition')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')

        plt.tight_layout()
        plt.show()

    def plot_humidity_vs_temperature(self, data=None):
        """Plot relationship between humidity and temperature."""
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return
            data = self.data

        if 'Humidity' not in data.columns or 'Temperature(°C)' not in data.columns:
            print("Humidity or Temperature columns not found in data.")
            return

        plt.figure(figsize=(10, 6))

        # Create scatter plot with regression line
        sns.regplot(x='Temperature(°C)', y='Humidity', data=data,
                    scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})

        plt.title('Relationship Between Temperature and Humidity')
        plt.xlabel('Temperature (°C)')
        plt.ylabel('Humidity')
        plt.grid(True, linestyle='--', alpha=0.7)

        # Calculate correlation coefficient
        corr = data['Temperature(°C)'].corr(data['Humidity'])
        plt.annotate(f'Correlation: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

        plt.tight_layout()
        plt.show()

    def plot_wind_rose(self, data=None, bins=16):
        """Plot wind rose diagram showing wind direction and speed distribution."""
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return
            data = self.data

        if 'Wind Bearing(degrees)' not in data.columns or 'Wind Speed(km/h)' not in data.columns:
            print("Wind Bearing or Wind Speed columns not found in data.")
            return

        # Create figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)

        # Convert wind direction to radians for polar plot
        wind_dir_rad = np.radians(data['Wind Bearing(degrees)'])
        wind_speed = data['Wind Speed(km/h)']

        # Create histogram of wind directions
        hist, bin_edges = np.histogram(
            wind_dir_rad, bins=bins, range=(0, 2*np.pi))

        # Normalize histogram
        width = 2*np.pi / bins
        bars = ax.bar(bin_edges[:-1], hist, width=width, bottom=0.0)

        # Set colors based on frequency
        norm = plt.Normalize(hist.min(), hist.max())
        for bar, h in zip(bars, hist):
            bar.set_facecolor(plt.cm.viridis(norm(h)))

        # Set direction labels
        directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                      'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
        angles = np.linspace(0, 2*np.pi, len(directions), endpoint=False)
        ax.set_thetagrids(np.degrees(angles), directions)

        # Customize chart
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)  # clockwise
        plt.title('Wind Rose Diagram\n(Wind Direction Distribution)', y=1.1)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label('Frequency')

        plt.tight_layout()
        plt.show()

        # Additionally, show wind speed statistics
        print("\nWind Speed Statistics:")
        print(f"Average Wind Speed: {wind_speed.mean():.2f} km/h")
        print(f"Maximum Wind Speed: {wind_speed.max():.2f} km/h")
        print(f"Minimum Wind Speed: {wind_speed.min():.2f} km/h")

    def plot_correlation_heatmap(self, data=None):
        """Plot correlation heatmap of weather variables."""
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return
            data = self.data

        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.shape[1] < 2:
            print("Need at least 2 numeric columns to create correlation heatmap.")
            return

        # Calculate correlation matrix
        corr_matrix = numeric_data.corr()

        plt.figure(figsize=(12, 10))

        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm',
                    vmin=-1, vmax=1, center=0, square=True, linewidths=.5,
                    cbar_kws={"shrink": .5}, fmt='.2f')

        plt.title('Correlation Between Weather Variables')
        plt.tight_layout()
        plt.show()

    def plot_visibility_analysis(self, data=None):
        """Analyze and plot visibility data."""
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return
            data = self.data

        if 'Visibility(km)' not in data.columns:
            print("Visibility column not found in data.")
            return

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: Visibility over time
        ax1.plot(data.index, data['Visibility(km)'], 'b-', alpha=0.7)
        ax1.set_title('Visibility Over Time')
        ax1.set_ylabel('Visibility (km)')
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Format x-axis dates
        date_range = (data.index.max() - data.index.min()).days
        if date_range <= 60:
            ax1.xaxis.set_major_formatter(DateFormatter('%b %d'))
            ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        else:
            ax1.xaxis.set_major_formatter(DateFormatter('%b %Y'))
            ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

        # Plot 2: Visibility distribution histogram
        sns.histplot(data['Visibility(km)'].dropna(),
                     bins=20, kde=True, ax=ax2)
        ax2.set_title('Visibility Distribution')
        ax2.set_xlabel('Visibility (km)')
        ax2.grid(True, linestyle='--', alpha=0.7)

        # Add visibility statistics as text
        stats_text = (f"Mean: {data['Visibility(km)'].mean():.2f} km\n"
                      f"Median: {data['Visibility(km)'].median():.2f} km\n"
                      f"Min: {data['Visibility(km)'].min():.2f} km\n"
                      f"Max: {data['Visibility(km)'].max():.2f} km")
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def analyze_pressure_trends(self, data=None):
        """Analyze and plot atmospheric pressure trends."""
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return
            data = self.data

        if 'Pressure(millibars)' not in data.columns:
            print("Pressure column not found in data.")
            return

        plt.figure(figsize=(12, 6))

        # Plot pressure over time
        plt.plot(data.index, data['Pressure(millibars)'], 'b-', alpha=0.7)

        # Add 5-day moving average
        rolling_avg = data['Pressure(millibars)'].rolling(
            window=5, center=True).mean()
        plt.plot(data.index, rolling_avg, 'r-',
                 linewidth=2, label='5-day Moving Average')

        # Calculate pressure change rate (derivative)
        pressure_change = data['Pressure(millibars)'].diff()

        # Mark significant pressure changes
        significant_drops = data.index[pressure_change < -3]
        significant_rises = data.index[pressure_change > 3]

        if len(significant_drops) > 0:
            plt.scatter(significant_drops, data.loc[significant_drops, 'Pressure(millibars)'],
                        color='blue', s=50, label='Significant Pressure Drop')

        if len(significant_rises) > 0:
            plt.scatter(significant_rises, data.loc[significant_rises, 'Pressure(millibars)'],
                        color='red', s=50, label='Significant Pressure Rise')

        plt.title('Atmospheric Pressure Trends')
        plt.ylabel('Pressure (millibars)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Format x-axis dates
        ax = plt.gca()
        date_range = (data.index.max() - data.index.min()).days
        if date_range <= 60:
            ax.xaxis.set_major_formatter(DateFormatter('%b %d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        else:
            ax.xaxis.set_major_formatter(DateFormatter('%b %Y'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        # Print pressure statistics and findings
        print("\nPressure Analysis Results:")
        print(
            f"Average Pressure: {data['Pressure(millibars)'].mean():.2f} millibars")
        print(
            f"Maximum Pressure: {data['Pressure(millibars)'].max():.2f} millibars on {data['Pressure(millibars)'].idxmax().strftime('%Y-%m-%d')}")
        print(
            f"Minimum Pressure: {data['Pressure(millibars)'].min():.2f} millibars on {data['Pressure(millibars)'].idxmin().strftime('%Y-%m-%d')}")
        print(
            f"Number of significant pressure drops: {len(significant_drops)}")
        print(
            f"Number of significant pressure rises: {len(significant_rises)}")

        # Check if there's a correlation between pressure and other weather events
        if 'Summary' in data.columns:
            low_pressure_conditions = data.loc[data['Pressure(millibars)'] < data['Pressure(millibars)'].quantile(
                0.25), 'Summary'].value_counts()
            high_pressure_conditions = data.loc[data['Pressure(millibars)'] > data['Pressure(millibars)'].quantile(
                0.75), 'Summary'].value_counts()

            print("\nWeather conditions during low pressure periods:")
            print(low_pressure_conditions)

            print("\nWeather conditions during high pressure periods:")
            print(high_pressure_conditions)

    def analyze_temperature_vs_time_of_day(self, data=None):
        """Analyze how temperature varies by time of day."""
        if data is None:
            if self.data is None:
                print("No data loaded. Please load data first.")
                return
            data = self.data

        if 'Temperature(°C)' not in data.columns:
            print("Temperature column not found in data.")
            return

        # Extract hour from the datetime index
        data_with_hour = data.copy()
        data_with_hour['Hour'] = data_with_hour.index.hour

        # Group by hour and calculate statistics
        hourly_temp = data_with_hour.groupby('Hour')['Temperature(°C)'].agg([
            'mean', 'min', 'max', 'std'])

        plt.figure(figsize=(12, 6))

        # Plot mean temperature by hour with error bands
        plt.plot(hourly_temp.index,
                 hourly_temp['mean'], 'b-', linewidth=2, label='Mean Temperature')
        plt.fill_between(hourly_temp.index,
                         hourly_temp['mean'] - hourly_temp['std'],
                         hourly_temp['mean'] + hourly_temp['std'],
                         alpha=0.2, color='blue', label='±1 Std Dev')

        # Plot min and max ranges
        plt.plot(hourly_temp.index,
                 hourly_temp['min'], 'g--', alpha=0.7, label='Minimum')
        plt.plot(hourly_temp.index,
                 hourly_temp['max'], 'r--', alpha=0.7, label='Maximum')

        plt.title('Temperature Variation by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Temperature (°C)')
        plt.xticks(range(0, 24))
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        # Annotate maximum and minimum temperature times
        max_hour = hourly_temp['mean'].idxmax()
        min_hour = hourly_temp['mean'].idxmin()

        plt.annotate(f'Peak: {hourly_temp.loc[max_hour, "mean"]:.1f}°C',
                     xy=(max_hour, hourly_temp.loc[max_hour, "mean"]),
                     xytext=(max_hour, hourly_temp.loc[max_hour, "mean"] + 1),
                     arrowprops=dict(arrowstyle='->'),
                     ha='center')

        plt.annotate(f'Low: {hourly_temp.loc[min_hour, "mean"]:.1f}°C',
                     xy=(min_hour, hourly_temp.loc[min_hour, "mean"]),
                     xytext=(min_hour, hourly_temp.loc[min_hour, "mean"] - 1),
                     arrowprops=dict(arrowstyle='->'),
                     ha='center')

        plt.tight_layout()
        plt.show()

        # Print findings
        print(
            f"Temperature typically peaks around {max_hour}:00 at {hourly_temp.loc[max_hour, 'mean']:.2f}°C")
        print(
            f"Temperature is typically lowest around {min_hour}:00 at {hourly_temp.loc[min_hour, 'mean']:.2f}°C")
        print(
            f"Daily temperature variation (max-min): {hourly_temp['mean'].max() - hourly_temp['mean'].min():.2f}°C")

    def comprehensive_analysis(self, start_date=None, end_date=None):
        """Run a comprehensive analysis on the weather data."""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return

        print("=" * 50)
        print("WEATHER DATA COMPREHENSIVE ANALYSIS")
        print("=" * 50)

        # 1. Filter by date range if specified
        if start_date or end_date:
            analysis_data = self.filter_by_date_range(start_date, end_date)
        else:
            analysis_data = self.data

        # 2. Basic statistics
        print("\n--- Basic Statistics ---")
        print(analysis_data.describe())

        # 3. Analyze weather conditions distribution
        print("\n--- Weather Conditions Distribution ---")
        if 'Summary' in analysis_data.columns:
            weather_counts = analysis_data['Summary'].value_counts()
            print(weather_counts)
            print("\nPlotting weather conditions distribution...")
            self.plot_weather_summary_distribution(analysis_data)

        # 4. Detect anomalies
        print("\n--- Anomaly Detection ---")
        if 'Temperature(°C)' in analysis_data.columns:
            print(analysis_data.columns)
            temp_anomalies = self.detect_anomalies(
                'Temperature(°C)', 2.5, analysis_data)
            if len(temp_anomalies) > 0:
                print("Temperature anomalies:")
                for date, value in temp_anomalies.items():
                    print(f"  {date.strftime('%Y-%m-%d')}: {value:.2f}°C")

        # 5. Monthly averages
        print("\n--- Monthly Averages ---")
        monthly_data = self.resample_data(analysis_data, 'M')
        print(monthly_data)

        # 6. Create visualizations
        print("\n--- Creating Visualizations ---")

        # Temperature over time with 7-day moving average
        print("Plotting temperature over time...")
        self.plot_temperature_over_time(analysis_data, rolling_window=7)

        # Humidity vs Temperature relationship
        print("Plotting humidity vs temperature relationship...")
        self.plot_humidity_vs_temperature(analysis_data)

        # Wind rose diagram
        print("Creating wind rose diagram...")
        self.plot_wind_rose(analysis_data)

        # Visibility analysis
        print("Analyzing visibility patterns...")
        self.plot_visibility_analysis(analysis_data)

        # Pressure trends
        print("Analyzing pressure trends...")
        self.analyze_pressure_trends(analysis_data)

        # Correlation heatmap
        print("Plotting correlation heatmap...")
        self.plot_correlation_heatmap(analysis_data)

        print("\nComprehensive analysis complete!")


# Example usage
def main():
    print("Weather Data Analytics Tool")
    print("--------------------------")

    # Initialize analyzer
    analyzer = WeatherDataAnalyzer()

    # Try to load from file
    import sys
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = "weather_data.csv"  # Default filename

    print(f"Loading data from {file_path}...")
    if analyzer.load_data(file_path):
        # Clean the data
        analyzer.clean_data()

        # Run comprehensive analysis
        analyzer.comprehensive_analysis()
    else:
        print("Failed to load data. Please check your file path.")


if __name__ == "__main__":
    main()
