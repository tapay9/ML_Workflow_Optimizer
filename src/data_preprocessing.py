import pandas as pd

df = pd.read_csv("data/hotel_bookings_updated_2024.csv")
# Display the first 5 rows of the dataset
print(df.head())
# Display dataset information
df.info()
# Display summary statistics of the dataset
print(df.describe())

# Check for missing values in the dataset
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Fill missing children values with 0
df["children"] = df["children"].fillna(0)

# Verify that there are no more missing values in the "children" column
print("\nMissing values in the 'children' column after filling:")
print(df["children"].isnull().sum())

# Fill missing country values with "Unknown"
df["country"] = df["country"].fillna("Unknown")

# Verify that there are no more missing values in the "country" column
print("\nMissing values in the 'country' column after filling:")
print(df["country"].isnull().sum())

# Fill missing agent values with 0
df["agent"] = df["agent"].fillna(0)

# Verify that there are no more missing values in the "agent" column
print("\nMissing values in the 'agent' column after filling:")
print(df["agent"].isnull().sum())

# Drop the company column
df = df.drop(columns=["company"])

# Verify that the company column was removed
print("\nIs 'company' still in the dataset?")
print("company" in df.columns)

print("\nTotal missing values in the dataset after cleaning:")
print(df.isnull().sum().sum())

# Check for duplicate rows
duplicate_count = df.duplicated().sum()
print("\nNumber of duplicate rows in the dataset:")
print(duplicate_count)

# Check the distribution of the target variable (assuming 'is_canceled' is the target)
print("\nCancellation distribution:")
print(df["is_canceled"].value_counts())

print("\nCancellation percentages:")
print(df["is_canceled"].value_counts(normalize=True) * 100)

# Create a total guests feature
df["total_guests"] = df["adults"] + df["children"] + df["babies"]

# Verify the new feature
print("\nTotal guests feature added:")
print(df[["adults", "children", "babies", "total_guests"]].head())

# Create a total nights feature
df["total_nights"] = df["stays_in_weekend_nights"] + df["stays_in_week_nights"]

# Verify the new feature
print("\nTotal nights feature added:")
print(df[["stays_in_weekend_nights", "stays_in_week_nights", "total_nights"]].head())

# Check for bookings with zero total guests
zero_guest_bookings = (df["total_guests"] == 0).sum()

print("\nBookings with zero total guests:")
print(zero_guest_bookings)

# Remove bookings with zero total guests
df = df[df["total_guests"] > 0].copy()

# Verify that zero total guest bookings have been removed
print("\nBookings with zero total guests removed.")
print((df["total_guests"] == 0).sum())

print("\nNumber of rows after removing zero guest bookings:")
print(len(df))

# Check for bookings with zero total nights
zero_night_bookings = (df["total_nights"] == 0).sum()

print("\nBookings with zero total nights:")
print(zero_night_bookings)

# Inspect bookings with zero total nights
zero_night_rows = df[df["total_nights"] == 0]

# quick inspection of zero-night bookings
print("\nSample of zero-night bookings:")
print(
    zero_night_rows[
        [
            "is_canceled",
            "adults",
            "children",
            "babies",
            "arrival_date_year",
            "arrival_date_month",
            "reservation_status"
        ]
    ].head(10)
)
# Remove columns that could cause data leakage
df = df.drop(columns=["reservation_status", "reservation_status_date"])

# Verify that the columns were removed
print("\nAre leakage columns still in the dataset?")
print("reservation_status" in df.columns)
print("reservation_status_date" in df.columns)

# Identify categorical columns in the dataset
categorical_columns = df.select_dtypes(include=["object"]).columns

print("\nCategorical columns in the dataset:")
print(categorical_columns)

print("\nUnique values in each categorical column:")

# Check the number of unique values in each categorical column
for column in categorical_columns:
    print(column, ":", df[column].nunique())

# Inspect hotel and city columns for possible redundancy
print("\nSample hotel and city values:")
print(df[["hotel", "city"]].head(20))

# Check the different hotel_types
df["hotel_type"] = df["hotel"].str.split(" - ").str[0]

print("\nHotel types:")
print(df["hotel_type"].value_counts())

# Drop the hotel original hotel column
df = df.drop(columns=["hotel"])

# Verify if hotel column was removed
print("\nIs hotel column still in the dataset?")
print("hotel" in df.columns)

# Check unique arrival years
print("\nUnique arrival years:")
print(df["arrival_date_year"].value_counts())

# Drop arrival_date because it has the same value
df = df.drop(columns=["arrival_date_year"])

#Verify that that column was dropped
print("\nIs arrival_date_year still in the dataset?")
print("arrival_date_year" in df.columns)

# Check the number of unique agents
print("\nUnique agents:")
print(df["agent"].nunique())

# Drop agent because it has too many unique values and could cause overfitting
df = df.drop(columns=["agent"])

# Verify that the agent column was dropped
print("\nIs agent column still in the dataset?")
print("agent" in df.columns)

print("\nRemaining columns:")
print(df.columns.tolist())

print("\nTotal number of columns remaining:")
print(len(df.columns))

print("\nPotentially time dependent columns:")

for column in ["assigned_room_type", "booking_changes", 
               "days_in_waiting_list"]:
    print(f"\n{column}:")
    print(df[column].value_counts().head())


# Drop assigned_room_type because it may contain information assigned after booking
df = df.drop(columns=["assigned_room_type"])

# Verify that the column was removed
print("\nIs assigned_room_type still in the dataset?")
print("assigned_room_type" in df.columns)

#Recheck categorical columns
categorical_columns = df.select_dtypes(include=["object"]).columns.tolist()

print("\nFinal categorical columns:")
print(categorical_columns)

print("\nNumber of categorical columns:")
print(len(categorical_columns)) 

print("\nIs is_canceled still in the dataset?")
print("is_canceled" in df.columns)

print("\nis_canceled distribution:")
print(df["is_canceled"].value_counts())

# Check the most common countries
print("\nMost common countries:")
print(df["country"].value_counts().head(15))

print("\nNumber of countries that appear fewer than 100 times:")
print((df["country"].value_counts() < 100).sum())

# Group rare countries into an "Other" category
country_counts = df["country"].value_counts()
rare_countries = country_counts[country_counts < 100].index

df["country"] = df["country"].replace(rare_countries, "Other")

# Check the number of country categories after grouping
print("\nNumber of country categories after grouping rare countries:")
print(df["country"].nunique())

print("\nUpdated country distribution:")
print(df["country"].value_counts().head(15))

# Check unique values for the remaining categorical columns
print("\nUnique values in remaining categorical columns:")

for column in categorical_columns:
    if column != "country":
        print(f"\n{column}:")
        print(df[column].value_counts())

# Final dataset checks
print("\n--- Final Dataset Check ---")

print("Dataset shape:", df.shape)
print("Total missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Target column present:", "is_canceled" in df.columns)

print("\nTarget distribution:")
print(df["is_canceled"].value_counts())

print("\nFinal columns:")
print(df.columns.tolist())

# Remove duplicate rows created after preprocessing
df = df.drop_duplicates().copy()

print("\nDuplicate rows after removal:")
print(df.duplicated().sum())

print("\nDataset shape after removing duplicates:")
print(df.shape)

df.to_csv("data/hotel_bookings_cleaned.csv", index=False)

print("\nCleaned dataset saved to 'data/hotel_bookings_cleaned.csv")

print("\nFinal is_canceled distribution:")
print(df["is_canceled"].value_counts())

print("\nFinal is_canceled percentages:")
print(df["is_canceled"].value_counts(normalize=True) * 100)
