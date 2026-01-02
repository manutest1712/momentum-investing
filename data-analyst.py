import pandas as pd
from sklearn.ensemble import RandomForestClassifier
df = pd.read_csv("Database_No_show_appointments/noshowappointments-kagglev2-may-2016.csv")


df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

df["No-show"] = df["No-show"].map({"Yes": 1, "No": 0})
df = df[df['Age'] >= 0]

df['WaitingDays'] = (
    df['AppointmentDay'].dt.normalize() -
    df['ScheduledDay'].dt.normalize()
).dt.days


# Appointment weekday
df["ApptWeekday"] = df["AppointmentDay"].dt.weekday

# Scheduled hour
df["ScheduledHour"] = df["ScheduledDay"].dt.hour

# Age buckets (non-linear effects)
df["AgeBucket"] = pd.cut(
    df["Age"],
    bins=[0, 18, 30, 45, 60, 120],
    labels=False
)



df.drop(columns=['PatientId', 'AppointmentID'], inplace=True)

print(df.describe())

df_enc = pd.get_dummies(
    df,
    columns=["Gender", "Neighbourhood"],
    drop_first=True
)




X = df_enc.drop(columns=["No-show", "AppointmentDay", "ScheduledDay"])
y = df_enc["No-show"]

rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
rf.fit(X, y)

feature_importance = (
    pd.Series(rf.feature_importances_, index=X.columns)
    .sort_values(ascending=False)
)

print(feature_importance.head(15))