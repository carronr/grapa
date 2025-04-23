# -*- coding: utf-8 -*-
"""
Created on Wed  Jan 17 12:53:00 2024

@author: Matthias Diethelm
Copyright (c) 2025, Empa, Laboratory for Thin Films and Photovoltaics, Romain Carron
"""

import numpy as np

from grapa.graph import Graph
from grapa.curve import Curve


class GraphMBE_TinyTusker(Graph):
    """
    Reads MBE Tiny Tusker .xlsx files exported from Wavelabs
    """

    FILEIO_GRAPHTYPE = "Small MBE Tiny Tusker data"

    AXISLABELS = [["Time", "t", "min"], ["Temperature", "T", "°C"]]

    # second axis

    @classmethod
    def isFileReadable(cls, _filename, fileext, line1="", line3="", **_kwargs):
        if fileext == ".csv" and line1.startswith("data label"):
            return True
        return False

    def readDataFromFile(self, attributes, **_kwargs):
        # Summary element
        self.data.append(Curve(np.vstack(([], [])), attributes))
        self[-1].update({"label": "Summary"})

        # determine header lines

        header = []
        with open(self.filename, "rb") as file:
            header_lines = 0
            for line in file:
                # Decoding the binary line to string
                decoded_line = line.decode("ascii").strip()
                # Splitting the line into elements
                elements = decoded_line.split(",")
                header.append(elements)
                if elements[0] == "timestamp":
                    break
                header_lines += 1

        header_labels = header[0]

        data_labels = [sub_list[0] for sub_list in header[1:-6]]
        objectIDs = [sub_list[1] for sub_list in header[1:-6]]
        object_titles = [sub_list[2] for sub_list in header[1:-6]]
        property_names = [sub_list[3] for sub_list in header[1:-6]]

        # read data
        # Specify the dtype for each column
        dtype = [("date", "M8[s]")] + [
            ("f" + str(i), "f8") for i in range(1, len(header[header_lines]))
        ]

        # Read the data from the file
        data = np.genfromtxt(
            self.filename,
            skip_header=header_lines + 1,
            delimiter=",",
            dtype=dtype,
            invalid_raise=False,
        )

        # Create an empty 2D array
        data_2d = np.empty((len(data), len(dtype)), dtype="O")

        # Fill the 2D array with data from the structured array
        for i in range(len(data)):
            for j in range(len(dtype)):
                data_2d[i, j] = data[i][j]

        # Convert the first np.datetime64 column to relative time

        file_start_date = data_2d[0, 0]
        for i in range(len(data_2d[:, 0])):
            data_2d[i, 0] = (
                (data_2d[i, 0] - file_start_date) / np.timedelta64(1, "s") / 60
            )

        data_2d = data_2d.astype(float)

        # Curve is standard xyCurve.

        datax = data_2d[:, 0]

        for i in range(len(data_2d[0, :]) - 1):
            datay = data_2d[:, i + 1]

            if objectIDs[i].startswith("shutter"):
                continue
            elif property_names[i] == "ActiveOut":
                self.data.append(Curve(np.vstack((datax, datay)), attributes))
                self.curve(-1).update(
                    {
                        "label": object_titles[i] + " ActiveOut",
                        "sample": attributes["label"].replace(" TT export", ""),
                        "ax_twinx": 1,
                    }
                )
            else:
                self.data.append(Curve(np.vstack((datax, datay)), attributes))
                self.curve(-1).update(
                    {
                        "label": object_titles[i] + " " + property_names[i],
                        "sample": attributes["label"].replace(" TT export", ""),
                    }
                )

        # graph cosmetics
        # self.update({'xlabel': self.formatAxisLabel(['Bias voltage', 'V', 'V']),
        #             'ylabel': self.formatAxisLabel(['Current density', 'J', 'mA cm$^{-2}$'])})
        # self.update({'axhline': [0, {'linewidth': 0.5}], 'axvline': [0, {'linewidth': 0.5}]})

        # self.data.append(Curve(np.transpose(data_2d[:,0:2]), attributes))
        self.headers.update({"collabels": ["Time [min]", "Temperature [°C]"]})
        self.update(
            {
                "xlabel": self.formatAxisLabel(GraphMBE_TinyTusker.AXISLABELS[0]),
                "ylabel": self.formatAxisLabel(GraphMBE_TinyTusker.AXISLABELS[1]),
            }
        )

        GraphMBE_TinyTusker.analyseData(
            self, attributes, property_names, object_titles, data_2d, file_start_date
        )

    def analyseData(
        self, attributes, property_names, object_titles, data_2d, file_start_date
    ):
        try:
            indices_activeout = [
                i for i, s in enumerate(property_names) if s == "ActiveOut"
            ]
            indices_pv = [i for i, s in enumerate(property_names) if s == "PV"]
            indices_targetsp = [
                i for i, s in enumerate(property_names) if s == "TargetSP"
            ]
            indices_open_request = [
                i for i, s in enumerate(property_names) if s == "openRequest"
            ]
            indices_close_request = [
                i for i, s in enumerate(property_names) if s == "closeRequest"
            ]
            indices_closed = [i for i, s in enumerate(property_names) if s == "closed"]
            indices_cubot = [
                i for i, s in enumerate(object_titles) if s == "Copper Bottom"
            ]
            indices_inbot = [
                i for i, s in enumerate(object_titles) if s == "Indium Bottom"
            ]
            indices_gabot = [
                i for i, s in enumerate(object_titles) if s == "Gallium Bottom"
            ]
            indices_sub = [
                i for i, s in enumerate(object_titles) if s == "Substrate Loop"
            ]
            indices_rbf_shutter = [
                i for i, s in enumerate(object_titles) if s == "Shutter RbF"
            ]
            indices_rbF = [i for i, s in enumerate(object_titles) if s == "RbF Loop"]
            indices_NaF = [i for i, s in enumerate(object_titles) if s == "NaF Loop"]

            targetSpLevels_cubot = GraphMBE_TinyTusker.targetSpLevels(
                data_2d[:, np.intersect1d(indices_cubot, indices_targetsp)[0] + 1]
            )
            targetSpLevels_inbot = GraphMBE_TinyTusker.targetSpLevels(
                data_2d[:, np.intersect1d(indices_inbot, indices_targetsp)[0] + 1]
            )
            targetSpLevels_gabot = GraphMBE_TinyTusker.targetSpLevels(
                data_2d[:, np.intersect1d(indices_gabot, indices_targetsp)[0] + 1]
            )
            targetSpLevels_sub = GraphMBE_TinyTusker.targetSpLevels(
                data_2d[:, np.intersect1d(indices_sub, indices_targetsp)[0] + 1]
            )
            targetSpLevels_NaF = GraphMBE_TinyTusker.targetSpLevels(
                data_2d[:, np.intersect1d(indices_NaF, indices_targetsp)[0] + 1]
            )

            print("The setpoint levels of Cu bot were: ", targetSpLevels_cubot)
            print("The setpoint levels of In bot were: ", targetSpLevels_inbot)
            print("The setpoint levels of Ga bot were: ", targetSpLevels_gabot)
            print("The setpoint levels of sub were: ", targetSpLevels_sub)
            print("The setpoint levels of NaF were: ", targetSpLevels_NaF)

            targetSpLevels_cubot_2ndstage, _ = max(
                enumerate(targetSpLevels_cubot), key=lambda x: x[1][1]
            )
            # Process start time

            datax = data_2d[:, 0]

            process_start_time = datax[
                np.argwhere(
                    data_2d[:, np.intersect1d(indices_sub, indices_activeout)[0] + 1]
                    > 0
                )[0]
            ][0]

            print("Start date and time of the loaded file:", file_start_date)
            print(
                "Start date and time of the process:",
                file_start_date + np.timedelta64(int(process_start_time * 60), "s"),
            )

            # 1st stage start time

            indices1 = [
                i for i, s in enumerate(object_titles) if s == "Substrate Shutter"
            ]

            try:
                main_shutter_opening_time = datax[
                    np.argwhere(
                        data_2d[
                            :, np.intersect1d(indices1, indices_open_request)[0] + 1
                        ]
                        > 0
                    )[0]
                ][0]
            except:
                main_shutter_opening_time = datax[
                    np.argwhere(
                        np.isnan(
                            data_2d[
                                :, np.intersect1d(indices1, indices_open_request)[0] + 1
                            ]
                        )
                        == 0
                    )[0]
                ][0]

            # Adjust time 0 of graph to main shutter opening time
            index = 0
            while self.curve(index) is not None:
                self.curve(index).setX(
                    self.curve(index).x() - main_shutter_opening_time
                )
                index += 1

            minus3_time_index = np.argmin(
                np.abs(datax - (main_shutter_opening_time - 3))
            )

            indices1 = [i for i, s in enumerate(object_titles) if s == "Copper Top"]
            cu_lip_t = data_2d[
                minus3_time_index, np.intersect1d(indices1, indices_pv)[0] + 1
            ]

            indices1 = [i for i, s in enumerate(object_titles) if s == "Indium Top"]
            in_lip_t = data_2d[
                minus3_time_index, np.intersect1d(indices1, indices_pv)[0] + 1
            ]

            indices1 = [i for i, s in enumerate(object_titles) if s == "Gallium Top"]
            ga_lip_t = data_2d[
                minus3_time_index, np.intersect1d(indices1, indices_pv)[0] + 1
            ]

            indices1 = [i for i, s in enumerate(object_titles) if s == "Selenium Loop"]
            se_t = data_2d[
                minus3_time_index, np.intersect1d(indices1, indices_pv)[0] + 1
            ]

            indices1 = [i for i, s in enumerate(object_titles) if s == "LN2"]
            ln2_t = data_2d[
                minus3_time_index, np.intersect1d(indices1, indices_pv)[0] + 1
            ]

            rbf_t = data_2d[
                minus3_time_index, np.intersect1d(indices_rbF, indices_pv)[0] + 1
            ]

            print(
                "Temperatures at process start (-3min): Cu lip:",
                "{:1.1f}".format(cu_lip_t),
                "°C In lip:",
                "{:1.1f}".format(in_lip_t),
                "°C Ga lip:",
                "{:1.1f}".format(ga_lip_t),
                "°C Se:",
                "{:1.1f}".format(se_t),
                "°C LN2:",
                "{:1.1f}".format(ln2_t),
                "°C RbF:",
                "{:1.1f}".format(rbf_t),
                "°C",
            )

            # 2nd stage start time

            start_time_2nd_stage = datax[
                targetSpLevels_cubot[targetSpLevels_cubot_2ndstage - 1][0] + 1
            ]

            # Stochiometric time (3rd stage start time)

            stoichiometric_point_time = datax[
                targetSpLevels_cubot[targetSpLevels_cubot_2ndstage][0] + 1
            ]

            print(
                "The stoichiometric point was at a time of",
                "{:1.1f}".format(stoichiometric_point_time - main_shutter_opening_time),
                "min",
            )

            # PDT start time

            start_time_pdt = datax[targetSpLevels_sub[2][0] + 1]

            try:
                rbf_t_shutter_opening_index = np.argwhere(
                    data_2d[
                        :,
                        np.intersect1d(indices_rbf_shutter, indices_open_request)[0]
                        + 1,
                    ]
                    > 0
                )[0]
            except:
                rbf_t_shutter_opening_index = np.argwhere(
                    np.isnan(
                        data_2d[
                            :,
                            np.intersect1d(indices_rbf_shutter, indices_open_request)[0]
                            + 1,
                        ]
                    )
                    == 0
                )[0]

            rbf_t_shutter_opening = data_2d[
                rbf_t_shutter_opening_index,
                np.intersect1d(indices_rbF, indices_pv)[0] + 1,
            ][0]

            try:
                rbf_t_shutter_closing_index = np.argwhere(
                    data_2d[
                        :,
                        np.intersect1d(indices_rbf_shutter, indices_close_request)[0]
                        + 1,
                    ]
                    > 0
                )[0]
            except:
                rbf_t_shutter_closing_index = np.argwhere(
                    data_2d[
                        :, np.intersect1d(indices_rbf_shutter, indices_closed)[0] + 1
                    ]
                    > 0
                )[0]
                print(
                    "RbF shutter closing request not visible in data, estimated from when closing message was confirmed"
                )

            rbf_t_shutter_closing = data_2d[
                rbf_t_shutter_closing_index,
                np.intersect1d(indices_rbF, indices_pv)[0] + 1,
            ][0]

            print(
                "RbF opening and closing temperatures:",
                "{:1.1f}".format(rbf_t_shutter_opening),
                "°C and",
                "{:1.1f}".format(rbf_t_shutter_closing),
                "°C",
            )

            # Ramp down start time

            start_time_ramp_down = datax[targetSpLevels_sub[3][0] + 1]

            start_time_list = [
                ["Ramp up", process_start_time],
                ["First stage", main_shutter_opening_time],
                ["Second stage", start_time_2nd_stage],
                ["Third stage", stoichiometric_point_time],
                ["PDT", start_time_pdt],
                ["Ramp down", start_time_ramp_down],
            ]

            # Check for deviation of the temperature values with respect to the set point

            for i in range(len(indices_pv)):
                if (
                    "Bottom" in object_titles[indices_pv[i]]
                    or "Substrate" in object_titles[indices_pv[i]]
                    or "NaF" in object_titles[indices_pv[i]]
                ):
                    deviation, stages = GraphMBE_TinyTusker.checkDeviation(
                        data_2d[:, indices_pv[i] + 1],
                        data_2d[:, indices_targetsp[i] + 1],
                        datax,
                        main_shutter_opening_time,
                        datax[rbf_t_shutter_closing_index] + 2,
                        start_time_list,
                    )
                    if deviation == True:
                        print(
                            "There was an deviation in the PV data of",
                            object_titles[indices_pv[i]],
                            "compared to its setpoint in the stages:",
                            stages,
                        )

            # Parameters for summary element, used for Openbis upload

            self.curve(0).update(
                {
                    "Timestamp Start": str(
                        file_start_date
                        + np.timedelta64(int(process_start_time * 60), "s")
                    ),
                    "Time stoichiometry": "{:1.1f}".format(
                        stoichiometric_point_time - main_shutter_opening_time
                    ),
                    "T substrate 1st stage": targetSpLevels_sub[1][1],
                    "T substrate 2nd stage": targetSpLevels_sub[2][1],
                    "T substrate 3rd stage": targetSpLevels_sub[2][1],
                    "T Cu source 2nd stage": targetSpLevels_cubot[
                        targetSpLevels_cubot_2ndstage
                    ][1],
                    "T subsrate PDT": targetSpLevels_sub[3][1],
                    "T NaF PDT": targetSpLevels_NaF[-1][1],
                    "T RbF PDT": "{:1.1f}".format(rbf_t_shutter_opening),
                }
            )

        except:
            print("There was an error in the analysis, check exported file format.")

    @staticmethod
    def targetSpLevels(data):
        last_occurrences = []
        last_value = None
        for i, val in enumerate(data):
            if val == last_value or (val in data[i + 1 :]):
                last_value = val
            elif last_value is not None:
                last_occurrences.append((i - 1, last_value))
                last_value = None
        if last_value is not None:
            last_occurrences.append((len(data) - 1, last_value))
        return np.array(last_occurrences, dtype=[("col1", int), ("col2", float)])

    @staticmethod
    def checkDeviation(pv, targetSP, datax, start_time, end_time, start_time_list):
        ind_start_time = np.argmin(np.abs(datax - start_time))
        ind_end_time = np.argmin(np.abs(datax - end_time))
        deviation_array = (
            np.abs(
                targetSP[ind_start_time:ind_end_time] - pv[ind_start_time:ind_end_time]
            )
            > 5
        )
        datax_temp = datax[ind_start_time:ind_end_time]
        deviation_times = datax_temp[deviation_array]

        stages = []
        if len(deviation_times) > 0:
            for i in range(len(start_time_list) - 1):
                stage, start_time = start_time_list[i]
                _, end_time = start_time_list[i + 1]
                if any(
                    start_time <= dev_time <= end_time for dev_time in deviation_times
                ):
                    stages.append(stage)
        return any(deviation_array), stages
