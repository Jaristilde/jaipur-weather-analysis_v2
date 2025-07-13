"""
Weather Data Analysis - Jaipur
Assignment Submission

This script demonstrates:
1. Loading CSV data into a pandas DataFrame
2. Displaying comprehensive data analysis
3. Creating various visualizations
4. Statistical analysis of weather patterns

Author: [Your Name]
Date: [Current Date]
"""

# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set up the plotting style for better visualizations
plt.style.use('default')
sns.set_palette("husl")

def load_and_analyze_data():
    """
    Load the CSV file and perform initial data analysis
    """
    print("=" * 60)
    print("WEATHER DATA ANALYSIS - JAIPUR")
    print("=" * 60)
    
    # Load the CSV file into a DataFrame
    dataframe = pd.read_csv('JaipurFinalCleanData.csv')
    
    # Display basic information
    print(f"\nDataset Shape: {dataframe.shape}")
    print(f"Date Range: {dataframe['date'].min()} to {dataframe['date'].max()}")
    print(f"Total Days: {len(dataframe)}")
    
    # Display first 10 rows
    print("\n" + "=" * 40)
    print("FIRST 10 ROWS OF DATA")
    print("=" * 40)
    print(dataframe.head(10))
    
    # Display last 5 rows
    print("\n" + "=" * 40)
    print("LAST 5 ROWS OF DATA")
    print("=" * 40)
    print(dataframe.tail())
    
    # Statistical summary
    print("\n" + "=" * 40)
    print("STATISTICAL SUMMARY")
    print("=" * 40)
    print(dataframe.describe())
    
    # Display data types and missing values
    print("\n" + "=" * 40)
    print("DATA TYPES AND MISSING VALUES")
    print("=" * 40)
    print("Data Types:")
    print(dataframe.dtypes)
    print(f"\nMissing Values:\n{dataframe.isnull().sum()}")
    
    return dataframe

def create_visualizations(dataframe):
    """
    Create comprehensive visualizations of the weather data
    """
    # Convert date column to datetime for better analysis
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    
    print("\n" + "=" * 40)
    print("CREATING VISUALIZATIONS")
    print("=" * 40)
    
    # Create main visualization figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Jaipur Weather Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Temperature trends over time
    axes[0, 0].plot(dataframe['date'], dataframe['mean_temperature'], 
                     label='Mean Temperature', color='red', linewidth=2)
    axes[0, 0].plot(dataframe['date'], dataframe['max_temperature'], 
                     label='Max Temperature', color='orange', linewidth=1.5, alpha=0.8)
    axes[0, 0].plot(dataframe['date'], dataframe['min_temperature'], 
                     label='Min Temperature', color='blue', linewidth=1.5, alpha=0.8)
    axes[0, 0].set_title('Temperature Trends Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Temperature (°C)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Humidity trends
    axes[0, 1].plot(dataframe['date'], dataframe['max_humidity'], 
                     label='Max Humidity', color='darkblue', linewidth=2)
    axes[0, 1].plot(dataframe['date'], dataframe['min_humidity'], 
                     label='Min Humidity', color='lightblue', linewidth=2)
    axes[0, 1].set_title('Humidity Trends Over Time')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Humidity (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Rainfall distribution
    axes[1, 0].hist(dataframe['rainfall'], bins=20, color='green', alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Rainfall Distribution')
    axes[1, 0].set_xlabel('Rainfall (mm)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Pressure trends
    axes[1, 1].plot(dataframe['date'], dataframe['mean_pressure'], 
                     color='purple', linewidth=2)
    axes[1, 1].set_title('Mean Pressure Over Time')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Pressure (hPa)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weather_analysis_main.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create additional analysis visualizations
    print("\nCreating additional analysis visualizations...")
    
    # Monthly temperature averages
    dataframe['month'] = dataframe['date'].dt.month
    monthly_temp = dataframe.groupby('month')['mean_temperature'].mean()
    
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Monthly temperature averages
    plt.subplot(2, 2, 1)
    monthly_temp.plot(kind='bar', color='coral')
    plt.title('Average Temperature by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Temperature (°C)')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    
    # Subplot 2: Temperature correlation heatmap
    plt.subplot(2, 2, 2)
    temp_cols = ['mean_temperature', 'max_temperature', 'min_temperature']
    correlation_matrix = dataframe[temp_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Temperature Correlation Matrix')
    
    # Subplot 3: Rainfall vs Temperature scatter plot
    plt.subplot(2, 2, 3)
    plt.scatter(dataframe['mean_temperature'], dataframe['rainfall'], 
                alpha=0.6, color='green')
    plt.xlabel('Mean Temperature (°C)')
    plt.ylabel('Rainfall (mm)')
    plt.title('Rainfall vs Temperature')
    
    # Subplot 4: Box plot of temperature by month
    plt.subplot(2, 2, 4)
    dataframe.boxplot(column='mean_temperature', by='month', ax=plt.gca())
    plt.title('Temperature Distribution by Month')
    plt.xlabel('Month')
    plt.ylabel('Temperature (°C)')
    
    plt.tight_layout()
    plt.savefig('weather_additional_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def display_summary_statistics(dataframe):
    """
    Display comprehensive summary statistics
    """
    print("\n" + "=" * 40)
    print("SUMMARY STATISTICS")
    print("=" * 40)
    
    print(f"Temperature Statistics:")
    print(f"  Mean Temperature: {dataframe['mean_temperature'].mean():.2f}°C")
    print(f"  Max Temperature: {dataframe['max_temperature'].max():.2f}°C")
    print(f"  Min Temperature: {dataframe['min_temperature'].min():.2f}°C")
    print(f"  Temperature Range: {dataframe['max_temperature'].max() - dataframe['min_temperature'].min():.2f}°C")
    
    print(f"\nRainfall Statistics:")
    print(f"  Total Rainfall: {dataframe['rainfall'].sum():.2f} mm")
    print(f"  Days with Rain: {(dataframe['rainfall'] > 0).sum()} days")
    print(f"  Rainy Days Percentage: {(dataframe['rainfall'] > 0).sum() / len(dataframe) * 100:.1f}%")
    print(f"  Average Rainfall (when it rains): {dataframe[dataframe['rainfall'] > 0]['rainfall'].mean():.2f} mm")
    print(f"  Maximum Daily Rainfall: {dataframe['rainfall'].max():.2f} mm")
    
    print(f"\nHumidity Statistics:")
    print(f"  Average Max Humidity: {dataframe['max_humidity'].mean():.2f}%")
    print(f"  Average Min Humidity: {dataframe['min_humidity'].mean():.2f}%")
    print(f"  Humidity Range: {dataframe['max_humidity'].max() - dataframe['min_humidity'].min():.2f}%")
    
    print(f"\nPressure Statistics:")
    print(f"  Average Pressure: {dataframe['mean_pressure'].mean():.2f} hPa")
    print(f"  Pressure Range: {dataframe['mean_pressure'].max() - dataframe['mean_pressure'].min():.2f} hPa")
    print(f"  Min Pressure: {dataframe['mean_pressure'].min():.2f} hPa")
    print(f"  Max Pressure: {dataframe['mean_pressure'].max():.2f} hPa")

def main():
    """
    Main function to execute the complete analysis
    """
    try:
        # Load and analyze data
        dataframe = load_and_analyze_data()
        
        # Create visualizations
        create_visualizations(dataframe)
        
        # Display summary statistics
        display_summary_statistics(dataframe)
        
        print("\n" + "=" * 40)
        print("ANALYSIS COMPLETE!")
        print("=" * 40)
        print("Files generated:")
        print("- weather_analysis_main.png")
        print("- weather_additional_analysis.png")
        print("\nThis script demonstrates:")
        print("1. Loading CSV data into pandas DataFrame")
        print("2. Comprehensive data exploration")
        print("3. Multiple visualization techniques")
        print("4. Statistical analysis of weather patterns")
        
    except FileNotFoundError:
        print("Error: JaipurFinalCleanData.csv file not found!")
        print("Please ensure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 