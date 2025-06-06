# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 10:57:14 2017

@author: Romain Carron
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""


from grapa.graph import Graph
from grapa.utils.graphIO import GraphIO
from grapa.mathModule import is_number


class GraphXLS(Graph):
    FILEIO_GRAPHTYPE = "Undetermined measurement type"

    @classmethod
    def isFileReadable(cls, _filename, fileext, **_kwargs):
        if fileext in [".xls", ".xlsx"]:
            return True
        return False

    def readDataFromFile(self, attributes, **_kwargs):
        try:
            from xlrd import open_workbook, biffh
        except ImportError as e:
            print("ImportError: cannot import xlrd. GraphXLS aborted.", e)
            return False

        try:
            wb = open_workbook(self.filename)
        except biffh.XLRDError as e:
            print(type(e), e)
            print("Abort reading the file")
            return
        sheet_id = attributes["complement"] if "complement" in attributes else 0
        if not isinstance(sheet_id, str) and not is_number(sheet_id):
            sheet_id = attributes["sheetid"] if "sheetid" in attributes else 0
        try:  # first try by name, then try by index, try first sheet by defaul
            sheet = wb.sheet_by_name(sheet_id)
        except Exception:
            try:
                sheet = wb.sheet_by_index(sheet_id)
            except Exception:
                sheet = wb.sheet_by_index(0)
                print(
                    'readDataFromFileXLS: spreadsheet "',
                    sheet_id,
                    '" not',
                    "found. Opened first sheet by default.",
                )
        number_of_rows = sheet.nrows
        number_of_columns = sheet.ncols
        items = []
        for row in range(0, number_of_rows):
            values = []
            for col in range(number_of_columns):
                try:  # required to process correctly rows with merged cells
                    value = sheet.cell(row, col).value
                except Exception:
                    value = ""
                try:
                    value = str(float(value))
                except ValueError:
                    pass
                finally:
                    values.append(value)
            items.append(values)
        # identify correct way to process data - is it database, or file generic ?
        func = GraphIO._funcReadDataFile(items, attributes)
        func(self, attributes, fileContent=items)
