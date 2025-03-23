from dotenv import load_dotenv
load_dotenv()
import pycountry
import random
import pandas as pd


def data_generation():
    # Define parameters
    num_campaigns = 50  # Number of unique campaign IDs
    num_countries_per_campaign = 50  # Each campaign advertised in at least 50 countries

    # Get a list of country names from pycountry
    all_countries = [country.alpha_2 for country in pycountry.countries]  # Get ISO 2-letter country codes

    # Define a list of random non-value added texts
    non_value_texts = [
        "httpsklsdfkljfj", "random_string_abc", "junk_value_xyz", "irrelevant_data_123", "unused_key_placeholder"
    ]

    # Generate data
    campaign_data = []
    for campaign_id in range(1, num_campaigns + 1):
        selected_countries = random.sample(all_countries, num_countries_per_campaign)  # Pick 50 random countries
        bid_values = {country: round(random.uniform(0.01, 0.1), 3) for country in selected_countries}  # Random bid values
        
        # Construct the bid_value dictionary with additional noise
        bid_value_with_noise = {
            random.choice(non_value_texts): random.choice(non_value_texts),  # Add random junk text
            "bid_value": bid_values,  # Actual bid values
            "bid_strategy": random.choice(["CPA", "CPC", "ROAS"])  # Random bidding strategy
        }
        
        for country in selected_countries:
            revenue = round(random.uniform(1000, 50000), 2)  # Random revenue between $1,000 and $50,000
            campaign_data.append([f"Campaign_{campaign_id}", country, revenue, bid_value_with_noise])

    # Convert to DataFrame
    df_campaigns = pd.DataFrame(campaign_data, columns=["Campaign_ID", "Country", "Revenue", "Bid_Raw"])

    # Display first 10 rows
    #print(df_campaigns.head(10))
    print(df_campaigns['Bid_Raw'][0])
    return df_campaigns

def data_generation2():
    # Define parameters
    num_campaigns = 50  # Number of unique campaign IDs
    num_countries_per_campaign = 50  # Each campaign advertised in at least 50 countries

    # Get a list of country names from pycountry
    all_countries = [country.alpha_2 for country in pycountry.countries]  # Get ISO 2-letter country codes

    # Define a list of random non-value added texts
    non_value_texts = [
        "httpsklsdfkljfj", "random_string_abc", "junk_value_xyz", "irrelevant_data_123", "unused_key_placeholder"
    ]

    # Define gender categories
    gender_categories = ["boy", "girl", "young people"]

    # Generate data
    campaign_data = []
    for campaign_id in range(1, num_campaigns + 1):
        selected_countries = random.sample(all_countries, num_countries_per_campaign)  # Pick 50 random countries
        bid_values = {country: round(random.uniform(0.01, 0.1), 3) for country in selected_countries}  # Small bid values
        
        # Construct the bid_value dictionary with additional noise
        bid_value_with_noise = {
            random.choice(non_value_texts): random.choice(non_value_texts),  # Add random junk text
            "bid_value": bid_values,  # Actual bid values
            "bid_strategy": random.choice(["CPA", "CPC", "ROAS"]),  # Random bidding strategy
        }

        # Create gender dictionary with same bid data
        gender_dict = {
            "gender": random.choice(gender_categories),
            **bid_value_with_noise  # Copy all keys from Bid_Raw
        }

        for country in selected_countries:
            revenue = round(random.uniform(1000, 50000), 2)  # Random revenue between $1,000 and $50,000
            campaign_data.append([
                f"Campaign_{campaign_id}", country, revenue, bid_value_with_noise, gender_dict
            ])

    # Convert to DataFrame
    df_campaigns = pd.DataFrame(campaign_data, columns=["Campaign_ID", "Country", "Revenue", "Bid_Raw", "Raw_Data"])

    # Display first 10 rows
    #print(df_campaigns.head(2))
    return df_campaigns

if __name__ == '__main__':
    data_generation2()
