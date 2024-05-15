
class SyntheticDataGenerator:
    def __init__(self):
        self.regions = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
        self.grades = ['A', 'B', 'C', 'D']
        self.prices = [100, 80, 60, 40]

    def create(self):
        data = {'Region': [], 'Grade': [], 'Price': []}
        for _ in range(100):
            data['Region'].append(np.random.choice(self.regions))
            data['Grade'].append(np.random.choice(self.grades))
            data['Price'].append(np.random.choice(self.prices))
        return pd.DataFrame(data)

    def save_to_csv(self, data, filename):
        data.to_csv(filename, index=False)

    def generate_and_save_data(self):
        data = self.create()
        self.save_to_csv(data, 'data.csv')
