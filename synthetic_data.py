
import pandas as pd
import numpy as np

class SyntheticDataGenerator:
    def __init__(self):
        self.num_customers = 100000
        self.num_transactions = 1000000

    def create(self):
        columns = ['customer_id', 'transaction_id', 'transaction_date', 'transaction_time', 'transaction_amount', 'transaction_type', 'transaction_location', 'transaction_status']
        data_types = [int, int, str, str, float, str, str, str]

        self.customer_id_distribution = np.random.randint(1, 1000001, self.num_customers)
        self.transaction_id_distribution = np.random.randint(1, 1000001, self.num_transactions)
        self.transaction_date_distribution = pd.date_range(start='2020-01-01', end='2022-12-31', periods=self.num_transactions)
        self.transaction_time_distribution = np.random.uniform(0, 24, self.num_transactions)
        self.transaction_amount_distribution = np.random.uniform(1, 1000, self.num_transactions)
        self.transaction_type_distribution = np.random.choice(['deposit', 'withdrawal', 'transfer'], self.num_transactions)
        self.transaction_location_distribution = np.random.choice(['online', 'branch', 'ATM'], self.num_transactions)
        self.transaction_status_distribution = ['success'] * 800000 + ['fraudulent'] * 200000

        self.data = {
            'customer_id': self.customer_id_distribution,
            'transaction_id': self.transaction_id_distribution,
            'transaction_date': [str(x) for x in self.transaction_date_distribution],
            'transaction_time': [str(x) for x in self.transaction_time_distribution],
            'transaction_amount': self.transaction_amount_distribution,
            'transaction_type': self.transaction_type_distribution,
            'transaction_location': self.transaction_location_distribution,
            'transaction_status': self.transaction_status_distribution
        }

    def save_to_csv(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)

    def generate_and_save_data(self, filename):
        self.create()
        self.save_to_csv(filename)