# ============================================================
# CODTECH INTERNSHIP - TASK 1
# API Integration and Data Visualization
# Author: [Your Name]
# Description: Fetches weather data from OpenWeatherMap API
#              for multiple Indian cities and creates a
#              comprehensive visualization dashboard.
# ============================================================

import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
API_KEY = "your_openweathermap_api_key"   # Replace with your key from openweathermap.org
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# List of Indian cities to analyze
CITIES = [
    "Davangere", "Bangalore", "Mumbai", "Delhi", "Chennai",
    "Hyderabad", "Kolkata", "Pune", "Jaipur", "Ahmedabad"
]

# ─────────────────────────────────────────
# STEP 1: FETCH WEATHER DATA FROM API
# ─────────────────────────────────────────
def fetch_weather(city):
    """Fetch current weather data for a given city."""
    params = {
        "q": f"{city},IN",
        "appid": API_KEY,
        "units": "metric"   # Celsius
    }
    try:
        response = requests.get(BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return {
            "City": city,
            "Temperature (°C)": data["main"]["temp"],
            "Feels Like (°C)": data["main"]["feels_like"],
            "Min Temp (°C)": data["main"]["temp_min"],
            "Max Temp (°C)": data["main"]["temp_max"],
            "Humidity (%)": data["main"]["humidity"],
            "Pressure (hPa)": data["main"]["pressure"],
            "Wind Speed (m/s)": data["wind"]["speed"],
            "Cloudiness (%)": data["clouds"]["all"],
            "Weather": data["weather"][0]["description"].title(),
        }
    except Exception as e:
        print(f"  [!] Could not fetch data for {city}: {e}")
        return None


def get_all_weather():
    """Fetch weather for all cities and return a DataFrame."""
    print("Fetching weather data for Indian cities...\n")
    records = []
    for city in CITIES:
        result = fetch_weather(city)
        if result:
            records.append(result)
            print(f"  ✓ {city}")
    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────
# STEP 2: USE SAMPLE DATA IF NO API KEY
# ─────────────────────────────────────────
def get_sample_data():
    """Returns realistic sample data for demonstration."""
    return pd.DataFrame({
        "City": CITIES,
        "Temperature (°C)": [31, 27, 33, 38, 36, 34, 30, 28, 40, 37],
        "Feels Like (°C)": [34, 29, 36, 42, 40, 37, 33, 31, 44, 41],
        "Min Temp (°C)": [28, 24, 30, 34, 33, 30, 27, 25, 36, 33],
        "Max Temp (°C)": [34, 30, 36, 41, 39, 37, 33, 31, 43, 40],
        "Humidity (%)": [65, 72, 80, 35, 50, 55, 78, 68, 20, 30],
        "Pressure (hPa)": [1008, 1012, 1006, 998, 1004, 1007, 1010, 1011, 995, 1000],
        "Wind Speed (m/s)": [3.5, 2.8, 5.2, 4.1, 3.9, 3.2, 4.5, 2.5, 6.1, 5.0],
        "Cloudiness (%)": [40, 60, 75, 10, 20, 30, 80, 55, 5, 15],
        "Weather": ["Partly Cloudy", "Overcast", "Rainy", "Clear Sky", "Hazy",
                    "Partly Cloudy", "Thunderstorm", "Cloudy", "Clear Sky", "Sunny"],
    })


# ─────────────────────────────────────────
# STEP 3: CREATE VISUALIZATIONS
# ─────────────────────────────────────────
def plot_dashboard(df):
    """Generate a comprehensive weather dashboard with 6 charts."""
    sns.set_theme(style="darkgrid", palette="muted")
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#1e1e2e")
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.5, wspace=0.4)

    title_color = "white"
    label_color = "#cccccc"

    # ── Chart 1: Temperature Bar Chart ──
    ax1 = fig.add_subplot(gs[0, :2])
    colors = sns.color_palette("RdYlGn_r", len(df))
    bars = ax1.bar(df["City"], df["Temperature (°C)"], color=colors, edgecolor="white", linewidth=0.5)
    ax1.plot(df["City"], df["Feels Like (°C)"], color="cyan", marker="o",
             linewidth=2, markersize=6, label="Feels Like (°C)")
    ax1.set_title("Current Temperature vs Feels Like", color=title_color, fontsize=13, fontweight="bold")
    ax1.set_ylabel("Temperature (°C)", color=label_color)
    ax1.tick_params(colors=label_color, rotation=30)
    ax1.set_facecolor("#2a2a3e")
    ax1.legend(facecolor="#2a2a3e", labelcolor="white")
    for bar, val in zip(bars, df["Temperature (°C)"]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val}°", ha="center", color="white", fontsize=8)

    # ── Chart 2: Humidity Pie Chart ──
    ax2 = fig.add_subplot(gs[0, 2])
    wedges, texts, autotexts = ax2.pie(
        df["Humidity (%)"], labels=df["City"], autopct="%1.0f%%",
        startangle=140, colors=sns.color_palette("Blues", len(df)),
        textprops={"fontsize": 7, "color": "white"}
    )
    ax2.set_title("Humidity Distribution (%)", color=title_color, fontsize=11, fontweight="bold")
    ax2.set_facecolor("#2a2a3e")

    # ── Chart 3: Wind Speed Horizontal Bar ──
    ax3 = fig.add_subplot(gs[1, 0])
    wind_sorted = df.sort_values("Wind Speed (m/s)", ascending=True)
    ax3.barh(wind_sorted["City"], wind_sorted["Wind Speed (m/s)"],
             color=sns.color_palette("coolwarm", len(df)))
    ax3.set_title("Wind Speed (m/s)", color=title_color, fontsize=11, fontweight="bold")
    ax3.tick_params(colors=label_color)
    ax3.set_facecolor("#2a2a3e")

    # ── Chart 4: Heatmap ──
    ax4 = fig.add_subplot(gs[1, 1:])
    heatmap_data = df[["City", "Temperature (°C)", "Humidity (%)",
                        "Wind Speed (m/s)", "Cloudiness (%)"]].set_index("City")
    sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlOrRd",
                ax=ax4, linewidths=0.5, linecolor="#1e1e2e",
                annot_kws={"size": 8})
    ax4.set_title("Weather Metrics Heatmap", color=title_color, fontsize=11, fontweight="bold")
    ax4.tick_params(colors=label_color, rotation=30)

    # ── Chart 5: Min-Max Temp Range ──
    ax5 = fig.add_subplot(gs[2, :2])
    x = range(len(df))
    ax5.fill_between(x, df["Min Temp (°C)"], df["Max Temp (°C)"],
                     alpha=0.4, color="orange", label="Temp Range")
    ax5.plot(x, df["Min Temp (°C)"], "b-o", markersize=5, label="Min Temp")
    ax5.plot(x, df["Max Temp (°C)"], "r-o", markersize=5, label="Max Temp")
    ax5.set_xticks(list(x))
    ax5.set_xticklabels(df["City"], rotation=30, color=label_color, fontsize=8)
    ax5.set_title("Daily Temperature Range", color=title_color, fontsize=11, fontweight="bold")
    ax5.set_ylabel("Temperature (°C)", color=label_color)
    ax5.legend(facecolor="#2a2a3e", labelcolor="white")
    ax5.set_facecolor("#2a2a3e")

    # ── Chart 6: Pressure vs Cloudiness Scatter ──
    ax6 = fig.add_subplot(gs[2, 2])
    scatter = ax6.scatter(df["Pressure (hPa)"], df["Cloudiness (%)"],
                          c=df["Temperature (°C)"], cmap="hot", s=100, edgecolors="white")
    for _, row in df.iterrows():
        ax6.annotate(row["City"], (row["Pressure (hPa)"], row["Cloudiness (%)"]),
                     fontsize=6.5, color="white", textcoords="offset points", xytext=(4, 4))
    plt.colorbar(scatter, ax=ax6, label="Temp (°C)")
    ax6.set_title("Pressure vs Cloudiness", color=title_color, fontsize=11, fontweight="bold")
    ax6.set_xlabel("Pressure (hPa)", color=label_color)
    ax6.set_ylabel("Cloudiness (%)", color=label_color)
    ax6.tick_params(colors=label_color)
    ax6.set_facecolor("#2a2a3e")

    # ── Main Title ──
    fig.suptitle(
        f"🌤  Indian Cities Weather Dashboard  |  {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        fontsize=16, fontweight="bold", color="white", y=0.98
    )

    plt.savefig("task1_weather_dashboard.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("\n✅ Dashboard saved as 'task1_weather_dashboard.png'")
    plt.show()


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Use real API or sample data
    if API_KEY == "your_openweathermap_api_key":
        print("ℹ  No API key found. Using sample data for demonstration.\n")
        df = get_sample_data()
    else:
        df = get_all_weather()

    print("\n📊 Weather Data Summary:")
    print(df[["City", "Temperature (°C)", "Humidity (%)", "Wind Speed (m/s)"]].to_string(index=False))

    # Save data to CSV
    df.to_csv("task1_weather_data.csv", index=False)
    print("\n✅ Data saved to 'task1_weather_data.csv'")

    # Generate dashboard
    plot_dashboard(df)
