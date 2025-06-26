import h5py  # 高性能HDF5存储
import os


class DataStorage:
    def __init__(self, filename="stock_data.h5"):
        self.filename = filename
        if not os.path.exists(self.filename):
            with h5py.File(self.filename, 'w') as f:
                f.create_group('stocks')

    def append_data(self, new_data):
        with h5py.File(self.filename, 'a') as f:
            timestamp = new_data['timestamp']
            grp = f['stocks'].create_group(timestamp)
            for stock, values in new_data.items():
                if stock != 'timestamp':
                    subgrp = grp.create_group(stock)
                    subgrp.create_dataset('price', data=values['price'])
                    subgrp.create_dataset('change', data=values['change'])