# import xlwt
# import xlrd
#
# book = xlwt.Workbook(encoding="utf-8")
#
# sheet1 = book.add_sheet("Sheet 1")
#
# sheet1.write(0, 0, "lol")
#
#
# sheet1.write(0, 1, 1)
# sheet1.write(1, 1, 2)
# sheet1.write(2, 1, 3)
# sheet1.write(5, 1, 4)
# sheet1.write(10, 1, 5)
#
# book.save("trial.xls")
#
# book = xlrd.open_workbook("trial.xls")
# sh = book.sheet_by_index(0)
# print(sh.ncols)
# for ry in range(sh.ncols):
#     col = (sh.col(ry))
#     for record in col:
#         print(record.value)

import tensorflow as tf
print(tf.config.list_physical_devices(device_type=None))