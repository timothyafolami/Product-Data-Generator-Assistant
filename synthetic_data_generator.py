
import pandas as pd
import numpy as np

class SyntheticDataGenerator:
    def __init__(self):
        pass

    def create(self):
        column_names = ['Student ID', 'Student Name', 'Date', 'Class', 'Attendance Status', 'Time of Attendance', 'Duration', 'Reason for Absence', 'Teacher', 'Classroom', 'Campus', 'Grade Level', 'Student Type']
        data_types = {'Student ID': 'int64', 'Student Name': 'str', 'Date': 'datetime64[ns]', 'Class': 'str', 'Attendance Status': 'category', 'Time of Attendance': 'datetime64[ns]', 'Duration': 'int64', 'Reason for Absence': 'str', 'Teacher': 'str', 'Classroom': 'str', 'Campus': 'str', 'Grade Level': 'int64', 'Student Type': 'category'}
        data_size = 500

        data = {
            'Student ID': np.random.randint(1, 1000, size=data_size),
            'Student Name': [f'Student {i}' for i in range(data_size)],
            'Date': pd.date_range(start='2022-01-01', periods=data_size),
            'Class': ['Math', 'Science', 'English', 'History'] * (data_size // 4),
            'Attendance Status': np.random.choice(['Present', 'Absent', 'Late', 'Tardy'], size=data_size),
            'Time of Attendance': pd.date_range(start='2022-01-01', periods=data_size),
            'Duration': np.random.randint(1, 100, size=data_size),
            'Reason for Absence': ['None'] * (data_size // 2) + ['Illness'] * (data_size // 4) + ['Family Emergency'] * (data_size // 4),
            'Teacher': ['Teacher 1', 'Teacher 2', 'Teacher 3'] * (data_size // 3),
            'Classroom': ['Room 101', 'Room 102', 'Room 103'] * (data_size // 3),
            'Campus': ['Main Campus', 'Satellite Campus', 'Online'] * (data_size // 3),
            'Grade Level': np.random.randint(1, 12, size=data_size),
            'Student Type': np.random.choice(['Undergraduate', 'Graduate', 'Alumni'], size=data_size)
        }
        return pd.DataFrame(data)

    def save_to_csv(self, data, filename):
        data.to_csv(filename, index=False)

    def generate_and_save_data(self, filename):
        data = self.create()
        self.save_to_csv(data, filename)