
import pandas as pd
import numpy as np
import datetime as dt

class SyntheticDataGenerator:
    def __init__(self):
        pass

    def create(self):
        column_names = ['Student ID', 'Date', 'Class/Subject', 'Attendance Status', 'Time of Attendance', 'Reason for Absence', 'Teacher/Instructor Name', 'Course Name', 'Semester/Trimester', 'Year']
        data_types = ['integer', 'datetime', 'string', 'string', 'string', 'string', 'string', 'string', 'string', 'integer']

        data = {
            'Student ID': np.random.randint(1, 1000, 1000),
            'Date': [dt.datetime(2022, np.random.randint(1, 13), np.random.randint(1, 28)) for _ in range(1000)],
            'Class/Subject': ['Math', 'English', 'Science', 'History'] * 250,
            'Attendance Status': ['Present', 'Absent', 'Tardy'] * 333,
            'Time of Attendance': [f"{np.random.randint(8, 17)}:00" for _ in range(1000)],
            'Reason for Absence': ['Sick', 'Family Emergency', 'Personal Appointment'] * 333,
            'Teacher/Instructor Name': ['Mr. Smith', 'Ms. Johnson', 'Dr. Lee'] * 333,
            'Course Name': ['Math 101', 'English 102', 'Science 103', 'History 104'] * 250,
            'Semester/Trimester': ['Fall', 'Spring', 'Summer'] * 333,
            'Year': np.random.randint(2020, 2025, 1000)
        }

        df = pd.DataFrame(data)

        df['Date'] = pd.to_datetime(df['Date'])

        return df

    def save_to_csv(self, df, filename):
        df.to_csv(filename, index=False)

    def generate_and_save_data(self):
        df = self.create()
        self.save_to_csv(df, 'attendance_record.csv')