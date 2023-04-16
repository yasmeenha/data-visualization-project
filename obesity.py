import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv("obesity.csv")

class Obesity:

    def obesity(self):
        # Obesity level distribution
        obesity_counts = df["NObeyesdad"].value_counts().reset_index()
        obesity_counts.columns = ["Obesity Level", "Count"]
        fig = px.bar(obesity_counts, x="Obesity Level", y="Count", title="Obesity Level Distribution")
        fig.update_layout(width=1000, height=500)
        return fig
    #visualizing obesity by age
    def age(self):
        age_by_obesity = df.groupby("NObeyesdad")["Age"].median().reset_index()
        fig = px.bar(age_by_obesity, x="NObeyesdad", y="Age", title="Median Age by Obesity Level", labels={"NObeyesdad": "Obesity Level", "Age": "Median Age (years)"})
        fig.update_layout(width=1000, height=500)
        return fig
    #visualizing obesity by height and specifying the dimensions for the chart
    def height(self):
        obesity_levels = df["NObeyesdad"].unique()
        height_by_obesity_level = []
        for obesity_level in obesity_levels:
            height_by_obesity_level.append(df[df["NObeyesdad"] == obesity_level]["Height"].mean())
        fig = px.bar(x=obesity_levels, y=height_by_obesity_level, color=obesity_levels, title="Height vs. Obesity Level",
                 labels={"x": "Obesity Level", "y": "Average Height (cm)", "color": "Obesity Level"})
        fig.update_layout(barmode="stack")
        fig.update_layout(width=1000, height=500)
        return fig
    ##visualizing obesity by weight and giving layout dimensions
    
    def weight(self):
        fig = px.histogram(df, x="Weight", color="NObeyesdad", nbins=30, title="Weight Distribution by Obesity Level", 
                           labels={"Weight": "Weight (kg)", "NObeyesdad": "Obesity Level"})
        fig.update_layout(width=1000, height=500)
        return fig
    ##visualizing obesity by family history and chart dimensions
    def fho(self):
        obesity_by_family_history = df.groupby(["NObeyesdad", "family_history_with_overweight"])["family_history_with_overweight"].count().unstack().reset_index()
        obesity_by_family_history.columns = ["Obesity Level", "No", "Yes"]
        fig = px.bar(obesity_by_family_history, x="Obesity Level", y=["No", "Yes"], title="Family History of Overweight vs. Obesity Level", 
                     barmode="stack")
        fig.update_layout(width=1000, height=500)
        return fig
    #visualizing obesity by high calorie food consumption and layout dimensions
    def favc(self):
        obesity_by_favc = df.groupby(["NObeyesdad", "FAVC"])["FAVC"].count().unstack().reset_index()
        obesity_by_favc.columns = ["Obesity Level", "No", "Yes"]
        fig = px.bar(obesity_by_favc, x=["No", "Yes"], y="Obesity Level", title="Frequent Consumption of High Caloric Food vs. Obesity Level",
                 barmode="stack", orientation="h", labels={"x": "Count", "y": "Obesity Level"})
        fig.update_layout(width=1000, height=500)
        return fig
    #visualizing obesity by frequency of consumption of high calorie food and layout dimensions
    def fcvc(self):
        fig = px.violin(df, x="NObeyesdad", y="FAVC", title="Frequency of consuming high caloric food vs. Obesity Level", 
                    labels={"NObeyesdad": "Obesity Level", "FAVC": "Frequency of consuming high caloric food"})
        fig.update_layout(width=1000, height=500)
        return fig
     #visualizing obesity by number of main meals 
    def ncp(self):
        fig = px.scatter(df, x="NCP", y="Weight", color="NObeyesdad", title="Number of Main Meals vs. Obesity Level", 
                     labels={"NCP": "Number of Main Meals", "Weight": "Weight in Kgs"})
        fig.update_layout(width=1000, height=500)
        return fig
#visualizing obesity by consumption of food between meals
    def caec(self):
        obesity_by_caec = df.groupby(["NObeyesdad", "CAEC"])["CAEC"].count().unstack().reset_index()
        obesity_by_caec.columns = ["Obesity Level", "No", "Sometimes", "Frequently", "Always"]
        fig = px.bar(obesity_by_caec, x="Obesity Level", y=["No", "Sometimes", "Frequently", "Always"], title="Consumption of food between meals vs. Obesity Level")
        fig.update_layout(width=1000, height=500)
        return fig
  #visualizing obesity by smoking habit  
    def smoke(self):
        obesity_by_smoke = df.groupby(["NObeyesdad", "SMOKE"])["SMOKE"].count().unstack().reset_index()
        obesity_by_smoke.columns = ["Obesity Level", "Non-Smoker", "Smoker"]
        fig = px.area(obesity_by_smoke, x="Obesity Level", y=["Non-Smoker", "Smoker"], title="Smoking Habit vs. Obesity Level", 
                  labels={"Obesity Level": "Obesity Level", "value": "Number of People"})
        fig.update_layout(barmode="stack", yaxis=dict(title="Number of People"))
        fig.update_layout(width=1000, height=500)
        return fig
    #visualizing obesity by water intake
    def water(self):
        df_grouped = df.groupby(["Gender", "NObeyesdad"], as_index=False).agg({"CH2O": "mean"})
        fig = px.bar(df_grouped, x="Gender", y="CH2O", color="NObeyesdad", title="Average Water Intake vs. Obesity Level by Gender", labels={"NObeyesdad": "Obesity Level", "CH2O": "Average Water Intake (liters)", "Gender": "Gender"})
        fig.update_layout(width=1000, height=500)
        return fig
    #visualizing obesity by selfcare capacity
    def calories(self):
        fig = px.bar(df, x="NObeyesdad", y="SCC", title="Self-care Capacity vs. Obesity Level", labels={"NObeyesdad": "Obesity Level", "SCC": "Self-care Capacity"})
        fig.update_layout(width=1000, height=500)
        return fig
    #visualizing obesity by physical activity frequency and bar chart layout
    def physical(self):
        fig = px.bar(df, x="FAF", y="NObeyesdad", color="Gender", title="Physical Activity Frequency vs. Obesity Level by Gender", labels={"NObeyesdad": "Obesity Level", "FAF": "Physical Activity Frequency"})
        fig.update_layout(width=1000, height=500)
        return fig
    
    def tue(self):
        # Aggregate the data by gender and obesity level
        agg_df = df.groupby(["Gender", "NObeyesdad"])["TUE"].mean().reset_index()
        # Create the bar chart
        fig = px.bar(agg_df, x="NObeyesdad", y="TUE", color="Gender", barmode="group", title="Mean Time Using Technology Devices vs. Obesity Level by Gender", labels={"NObeyesdad": "Obesity Level", "TUE": "Mean Time Using Technology Devices (hours)"})
        fig.update_layout(width=1000, height=500)
        return fig
    #visualizing obesity by alcohol intake
    def alcohol(self):
        obesity_by_calc = df.groupby(["NObeyesdad", "CALC"])["CALC"].count().unstack().reset_index()
        obesity_by_calc.columns = ["Obesity Level", "No", "Sometimes", "Frequently", "Always"]
        fig = px.bar(obesity_by_calc, x="Obesity Level", y=["No", "Sometimes", "Frequently", "Always"], title="Calories Intake Monitoring vs. Obesity Level", barmode="stack")
        fig.update_layout(width=1000, height=500)
        return fig
    #visualizing obesity by public transportation
    def mtrans(self):
        obesity_by_mtrans = df.groupby(["NObeyesdad", "MTRANS"])["MTRANS"].count().unstack().reset_index()
        obesity_by_mtrans.columns = ["Obesity Level", "Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"]
        fig = px.bar(obesity_by_mtrans, x="Obesity Level", y=["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"], title="Transportation Mode vs. Obesity Level", barmode="stack")
        fig.update_layout(width=1000, height=500)
        return fig
