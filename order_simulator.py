import numpy as np
import pandas as pd
import math
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings


# Constants
ORDER_TYPE_SPORADIC = 0
ORDER_TYPE_WEEKLY = 1
ORDER_TYPE_BIWEEKLY = 2

ORDER_FREQ_MAIN_DAY = 1
ORDER_FREQ_TWO_DAYS = 2
ORDER_FREQ_ALL_DAYS = 3


def sample_from_gaussian_mixture_model(components, num_samples=1):
    """Sample from Gaussian Mixture Model."""
    weights = [c[2] for c in components]
    component_indices = np.random.choice(np.arange(len(components)), size=num_samples, p=weights)

    samples = []
    for idx in component_indices:
        mean, std_deviation, _ = components[idx]
        sample = np.random.normal(mean, std_deviation)
        samples.append(sample)

    return np.array(samples)


def sample_discrete_normal(mean, std_deviation, num_samples=1):
    """Sample from discrete normal distribution."""
    continuous_samples = np.random.normal(mean, std_deviation, num_samples)
    discrete_samples = np.round(continuous_samples).astype(int)
    return discrete_samples


def order_around_position(arr, position):
    """Order array around a specific position."""
    if position < 0 or position >= len(arr):
        raise ValueError("Position is out of range.")

    arr.sort(reverse=True)
    ordered_array = arr.copy()
    ordered_array[position] = arr[0]

    idx = 1
    left, right = position - 1, position + 1

    while left >= 0 or right < len(arr):
        if left >= 0:
            ordered_array[left] = arr[idx]
            left -= 1
            idx += 1

        if right < len(arr):
            ordered_array[right] = arr[idx]
            right += 1
            idx += 1

        if idx >= len(arr):
            break

    return ordered_array


def adjust_unittype_id(chosen_unittype, unit_type, permissable_unittypes):
    """Adjust unit type to match permissible types."""
    if chosen_unittype == unit_type:
        return chosen_unittype

    dev = chosen_unittype - unit_type
    idx_prefered_unit = permissable_unittypes.index(unit_type)

    if idx_prefered_unit + dev < len(permissable_unittypes):
        chosen_unittype = np.array(permissable_unittypes)[idx_prefered_unit + dev]
        return chosen_unittype

    chosen_unittype = np.array(permissable_unittypes)[idx_prefered_unit + dev - len(permissable_unittypes)]
    return chosen_unittype


def sample_orders_from_customer_profiles(customer_profiles, weekly_variation, even_week=0,
                                         allowed_unit_types=[10, 18, 20, 22, 24, 30, 40, 80],
                                         sample_time=False):
    """Sample orders from customer profiles."""
    sampled_orders = []

    for profile in customer_profiles:
        total_volume = profile['Total_Volume']
        num_connections = profile['Number_of_Connections']
        customer_id = profile['Customer_ID']

        for connection in profile['Connections']:
            if customer_id == "excluded_customers":
                if num_connections == 0 or total_volume == 0:
                    continue
            else:
                if connection['Order_Frequency'] == ORDER_TYPE_SPORADIC:
                    if np.random.rand(1) < connection['Order_Probability']:
                        continue
                elif connection['Order_Frequency'] == ORDER_TYPE_BIWEEKLY:
                    if connection['Order_Probability'] != even_week:
                        continue

            connection_id = connection['Connection_ID']
            order_distribution = connection['Order_Distribution']
            preferred_days = connection['Preferred_Days']
            weekly_volume = connection['Max_Orders_Per_Day']
            unit_type = connection['Unit_Type']
            unit_type_variation = connection['Unit_Type_Std_Dev']
            weight_distribution = connection['Weight_Distribution_Components']

            # Determine distribution of orders across the week
            if order_distribution == ORDER_FREQ_MAIN_DAY:
                daily_order_volumes = np.random.rand(5).tolist()
                daily_order_volumes = np.array(order_around_position(daily_order_volumes, preferred_days - 1))
                daily_order_volumes = np.concatenate((daily_order_volumes, np.random.normal(0.2, 0.05, 2)))

            elif order_distribution == ORDER_FREQ_TWO_DAYS:
                daily_order_volumes = np.random.rand(5).tolist()
                daily_order_volumes = (np.array(order_around_position(daily_order_volumes, preferred_days[0] - 1)) +
                                       np.array(order_around_position(daily_order_volumes, preferred_days[1] - 1))) / 2
                daily_order_volumes = np.concatenate((daily_order_volumes, np.random.normal(0.2, 0.05, 2)))

            else:
                daily_order_volumes = np.ones(5) - np.random.normal(0.2, 0.1, 5)
                daily_order_volumes = np.concatenate((daily_order_volumes, np.random.normal(0.2, 0.05, 2)))

            # Adjust volumes based on weekly variation
            daily_order_volumes = np.clip(daily_order_volumes * (weekly_volume + int(weekly_volume * weekly_variation)),
                                          0, (weekly_volume + int(weekly_volume * weekly_variation)))

            # Sample orders for each day
            for day, daily_orders in enumerate(daily_order_volumes, start=1):
                for _ in range(math.ceil(daily_orders)):
                    weight = sample_from_gaussian_mixture_model(weight_distribution)
                    weight = np.clip(weight, 3, None)

                    chosen_unit_type = sample_discrete_normal(unit_type, unit_type_variation, 1)
                    chosen_unit_type = adjust_unittype_id(chosen_unit_type, unit_type, allowed_unit_types)

                    if sample_time:
                        arrival_time = connection['Arrival_Hour'] + np.random.randint(0, 2)
                        delivery_deadline = connection['Delivery_Deadline'] + np.random.randint(0, 2)
                        sampled_orders.append((customer_id, connection_id, weight[0], chosen_unit_type[0], day,
                                               arrival_time, delivery_deadline))
                    else:
                        sampled_orders.append((customer_id, connection_id, weight[0], chosen_unit_type[0], day))

    return sampled_orders


def sample_orders_from_connection_profiles(connection_profiles, weekly_variation, even_week=0,
                                           allowed_unit_types=[10, 18, 20, 22, 24, 30, 40, 80],
                                           sample_time=False):
    """Sample orders from connection profiles."""
    sampled_orders = []

    for profile in connection_profiles:
        connection_id = profile['Connection_ID']
        volume_per_week = profile['Max_Orders_Per_Day']
        unit_type = profile['Unit_Type']
        unit_type_variation = profile['Unit_Type_Std_Dev']
        weight_distribution = profile['Weight_Distribution_Components']
        order_frequency_pattern = profile['Order_Frequency']

        if volume_per_week == 0:
            continue

        dummy_customer_id = 'dummy_customer'

        # Determine distribution of orders across the week
        if order_frequency_pattern == ORDER_FREQ_MAIN_DAY:
            daily_order_volumes = np.maximum(np.random.rand(5), 0.3).tolist()
            daily_order_volumes = np.array(order_around_position(daily_order_volumes, profile['Preferred_Days'] - 1))
            daily_order_volumes = np.concatenate((daily_order_volumes, np.random.normal(0.2, 0.05, 2)))

        elif order_frequency_pattern == ORDER_FREQ_TWO_DAYS:
            daily_order_volumes = np.maximum(np.random.rand(5), 0.3).tolist()
            day_1_distribution = np.array(order_around_position(daily_order_volumes, profile['Preferred_Days'][0] - 1))
            day_2_distribution = np.array(order_around_position(daily_order_volumes, profile['Preferred_Days'][1] - 1))

            daily_order_volumes = np.where(np.isin(range(5), profile['Preferred_Days'] - 1),
                                           np.maximum(day_1_distribution, day_2_distribution),
                                           (day_1_distribution + day_2_distribution) / 2)
            daily_order_volumes = np.concatenate((daily_order_volumes, np.random.normal(0.2, 0.05, 2)))

        else:
            daily_order_volumes = np.maximum(np.ones(5) - np.random.normal(0.2, 0.1, 5), 0.3)
            daily_order_volumes = np.concatenate((daily_order_volumes, np.random.normal(0.2, 0.05, 2)))

        # Adjust volumes based on weekly variation
        daily_order_volumes = np.clip(daily_order_volumes * (volume_per_week + int(volume_per_week * weekly_variation)),
                                      0, (volume_per_week + int(volume_per_week * weekly_variation)))

        # Sample orders for each day
        for day, daily_orders in enumerate(daily_order_volumes, start=1):
            for _ in range(math.ceil(daily_orders)):
                weight = sample_from_gaussian_mixture_model(weight_distribution)
                weight = np.clip(weight, 3, None)

                chosen_unit_type = sample_discrete_normal(unit_type, unit_type_variation, 1)
                chosen_unit_type = adjust_unittype_id(chosen_unit_type, unit_type, allowed_unit_types)

                if sample_time:
                    arrival_time = profile.get('Arrival_Hour', 9) + np.random.randint(0, 2)
                    delivery_deadline = profile.get('Delivery_Deadline', 17) + np.random.randint(0, 2)
                    sampled_orders.append((dummy_customer_id, connection_id, weight[0], chosen_unit_type[0], day,
                                           arrival_time, delivery_deadline))
                else:
                    sampled_orders.append((dummy_customer_id, connection_id, weight[0], chosen_unit_type[0], day))

    return sampled_orders


def generate_order_df(profiles, UT_list, sim_weeks=10, sample_time=False, sample_from_connections=False):
    """
    Generate order dataframe from profiles.

    Parameters:
    -----------
    profiles : list
        List of customer or connection profiles
    UT_list : list
        List of allowed unit types
    sim_weeks : int
        Number of weeks to simulate
    sample_time : bool
        Whether to sample arrival and delivery times
    sample_from_connections : bool
        If True, use connection profiles; if False, use customer profiles

    Returns:
    --------
    pd.DataFrame
        DataFrame with sampled orders
    """
    general_df = pd.DataFrame()

    for week in range(sim_weeks):
        weekly_variation = 0.01
        even_week = week % 2

        # Choose appropriate sampling function
        if sample_from_connections:
            current_orders = sample_orders_from_connection_profiles(
                profiles, weekly_variation, even_week=even_week,
                allowed_unit_types=UT_list, sample_time=sample_time
            )
        else:
            current_orders = sample_orders_from_customer_profiles(
                profiles, weekly_variation, even_week=even_week,
                allowed_unit_types=UT_list, sample_time=sample_time
            )

        # Construct DataFrame
        if sample_time:
            df_orders = pd.DataFrame(current_orders,
                                     columns=['Customer_id', 'Connection', 'Weight', 'Unit_Type',
                                              'Day', 'Arrival Time', 'Delivery Deadline'])
        else:
            df_orders = pd.DataFrame(current_orders,
                                     columns=['Customer_id', 'Connection', 'Weight', 'Unit_Type', 'Day'])

        # Add week information
        df_orders['Week'] = week
        general_df = pd.concat([general_df, df_orders], ignore_index=True)

    return general_df


# Example usage with sample data
if __name__ == "__main__":
    # Create sample customer profile
    sample_profiles = [
        {
            'Customer_ID': 'CUST001',
            'Total_Volume': 100,
            'Number_of_Connections': 1,
            'Connections': [
                {
                    'Connection_ID': 1,
                    'Order_Distribution': ORDER_FREQ_MAIN_DAY,
                    'Order_Frequency': ORDER_TYPE_WEEKLY,
                    'Order_Probability': 1.0,
                    'Preferred_Days': 2,  # Tuesday
                    'Max_Orders_Per_Day': 5,
                    'Unit_Type': 20,
                    'Unit_Type_Std_Dev': 0.5,
                    'Weight_Distribution_Components': [(15.0, 2.0, 0.6), (25.0, 3.0, 0.4)],
                    'Arrival_Hour': 9,
                    'Delivery_Deadline': 17
                }
            ]
        }
    ]

    # Unit types list
    unit_types = [10, 18, 20, 22, 24, 30, 40, 80]

    # Generate orders for 2 weeks
    orders_df = generate_order_df(
        profiles=sample_profiles,
        UT_list=unit_types,
        sim_weeks=2,
        sample_time=False,
        sample_from_connections=False
    )

    print("Generated Orders:")
    print(orders_df)
    print(f"\nTotal orders generated: {len(orders_df)}")