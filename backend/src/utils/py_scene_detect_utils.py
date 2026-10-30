"""
BSD 3-Clause License

Copyright (C) 2024, Brandon Castellano

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy
import math
import typing as ty
from enum import Enum
import cv2
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, List, Optional, Any, Dict, Iterable, Set, TextIO, Union

import csv
import os.path
from pathlib import Path

MAX_FPS_DELTA: float = 1.0 / 100000
"""Maximum amount two framerates can differ by for equality testing."""

_SECONDS_PER_MINUTE = 60.0
_SECONDS_PER_HOUR = 60.0 * _SECONDS_PER_MINUTE
_MINUTES_PER_HOUR = 60.0


class FrameTimecode:
    """Object for frame-based timecodes, using the video framerate to compute back and
    forth between frame number and seconds/timecode.

    A timecode is valid only if it complies with one of the following three types/formats:
        1. Timecode as `str` in the form "HH:MM:SS[.nnn]" (`"01:23:45"` or `"01:23:45.678"`)
        2. Number of seconds as `float`, or `str` in form  "SSSS.nnnn" (`"45.678"`)
        3. Exact number of frames as `int`, or `str` in form NNNNN (`456` or `"456"`)
    """

    def __init__(
        self,
        timecode: Union[int, float, str, "FrameTimecode"] = None,
        fps: Union[int, float, str, "FrameTimecode"] = None,
    ):
        """
        Arguments:
            timecode: A frame number (int), number of seconds (float), or timecode (str in
                the form `'HH:MM:SS'` or `'HH:MM:SS.nnn'`).
            fps: The framerate or FrameTimecode to use as a time base for all arithmetic.
        Raises:
            TypeError: Thrown if either `timecode` or `fps` are unsupported types.
            ValueError: Thrown when specifying a negative timecode or framerate.
        """
        # The following two properties are what is used to keep track of time
        # in a frame-specific manner.  Note that once the framerate is set,
        # the value should never be modified (only read if required).
        # TODO(v1.0): Make these actual @properties.
        self.framerate = None
        self.frame_num = None

        # Copy constructor.  Only the timecode argument is used in this case.
        if isinstance(timecode, FrameTimecode):
            self.framerate = timecode.framerate
            self.frame_num = timecode.frame_num
            if fps is not None:
                raise TypeError(
                    "Framerate cannot be overwritten when copying a FrameTimecode."
                )
        else:
            # Ensure other arguments are consistent with API.
            if fps is None:
                raise TypeError("Framerate (fps) is a required argument.")
            if isinstance(fps, FrameTimecode):
                fps = fps.framerate

            # Process the given framerate, if it was not already set.
            if not isinstance(fps, (int, float)):
                raise TypeError("Framerate must be of type int/float.")
            if (isinstance(fps, int) and not fps > 0) or (
                isinstance(fps, float) and not fps >= MAX_FPS_DELTA
            ):
                raise ValueError("Framerate must be positive and greater than zero.")
            self.framerate = float(fps)

        # Process the timecode value, storing it as an exact number of frames.
        if isinstance(timecode, str):
            self.frame_num = self._parse_timecode_string(timecode)
        else:
            self.frame_num = self._parse_timecode_number(timecode)

    # TODO(v1.0): Add a `frame` property to replace the existing one and deprecate this getter.
    def get_frames(self) -> int:
        """Get the current time/position in number of frames.  This is the
        equivalent of accessing the self.frame_num property (which, along
        with the specified framerate, forms the base for all of the other
        time measurement calculations, e.g. the :meth:`get_seconds` method).

        If using to compare a :class:`FrameTimecode` with a frame number,
        you can do so directly against the object (e.g. ``FrameTimecode(10, 10.0) <= 10``).

        Returns:
            int: The current time in frames (the current frame number).
        """
        return self.frame_num

    # TODO(v1.0): Add a `framerate` property to replace the existing one and deprecate this getter.
    def get_framerate(self) -> float:
        """Get Framerate: Returns the framerate used by the FrameTimecode object.

        Returns:
            float: Framerate of the current FrameTimecode object, in frames per second.
        """
        return self.framerate

    def equal_framerate(self, fps) -> bool:
        """Equal Framerate: Determines if the passed framerate is equal to that of this object.

        Arguments:
            fps: Framerate to compare against within the precision constant defined in this module
                (see :data:`MAX_FPS_DELTA`).
        Returns:
            bool: True if passed fps matches the FrameTimecode object's framerate, False otherwise.

        """
        return math.fabs(self.framerate - fps) < MAX_FPS_DELTA

    # TODO(v1.0): Add a `seconds` property to replace this and deprecate the existing one.
    def get_seconds(self) -> float:
        """Get the frame's position in number of seconds.

        If using to compare a :class:`FrameTimecode` with a frame number,
        you can do so directly against the object (e.g. ``FrameTimecode(10, 10.0) <= 1.0``).

        Returns:
            float: The current time/position in seconds.
        """
        return float(self.frame_num) / self.framerate

    # TODO(v1.0): Add a `timecode` property to replace this and deprecate the existing one.
    def get_timecode(self, precision: int = 3, use_rounding: bool = True) -> str:
        """Get a formatted timecode string of the form HH:MM:SS[.nnn].

        Args:
            precision: The number of decimal places to include in the output ``[.nnn]``.
            use_rounding: Rounds the output to the desired precision. If False, the value
                will be truncated to the specified precision.

        Returns:
            str: The current time in the form ``"HH:MM:SS[.nnn]"``.
        """
        # Compute hours and minutes based off of seconds, and update seconds.
        secs = self.get_seconds()
        hrs = int(secs / _SECONDS_PER_HOUR)
        secs -= hrs * _SECONDS_PER_HOUR
        mins = int(secs / _SECONDS_PER_MINUTE)
        secs = max(0.0, secs - (mins * _SECONDS_PER_MINUTE))
        if use_rounding:
            secs = round(secs, precision)
        secs = min(_SECONDS_PER_MINUTE, secs)
        # Guard against emitting timecodes with 60 seconds after rounding/floating point errors.
        if int(secs) == _SECONDS_PER_MINUTE:
            secs = 0.0
            mins += 1
            if mins >= _MINUTES_PER_HOUR:
                mins = 0
                hrs += 1
        # We have to extend the precision by 1 here, since `format` will round up.
        msec = format(secs, ".%df" % (precision + 1)) if precision else ""
        # Need to include decimal place in `msec_str`.
        msec_str = msec[-(2 + precision) : -1]
        secs_str = f"{int(secs):02d}{msec_str}"
        # Return hours, minutes, and seconds as a formatted timecode string.
        return "%02d:%02d:%s" % (hrs, mins, secs_str)

    # TODO(v1.0): Add a `previous` property to replace the existing one and deprecate this getter.
    def previous_frame(self) -> "FrameTimecode":
        """Return a new FrameTimecode for the previous frame (or 0 if on frame 0)."""
        new_timecode = FrameTimecode(self)
        new_timecode.frame_num = max(0, new_timecode.frame_num - 1)
        return new_timecode

    def _seconds_to_frames(self, seconds: float) -> int:
        """Convert the passed value seconds to the nearest number of frames using
        the current FrameTimecode object's FPS (self.framerate).

        Returns:
            Integer number of frames the passed number of seconds represents using
            the current FrameTimecode's framerate property.
        """
        return round(seconds * self.framerate)

    def _parse_timecode_number(self, timecode: Union[int, float]) -> int:
        """Parse a timecode number, storing it as the exact number of frames.
        Can be passed as frame number (int), seconds (float)

        Raises:
            TypeError, ValueError
        """
        # Process the timecode value, storing it as an exact number of frames.
        # Exact number of frames N
        if isinstance(timecode, int):
            if timecode < 0:
                raise ValueError(
                    "Timecode frame number must be positive and greater than zero."
                )
            return timecode
        # Number of seconds S
        elif isinstance(timecode, float):
            if timecode < 0.0:
                raise ValueError(
                    "Timecode value must be positive and greater than zero."
                )
            return self._seconds_to_frames(timecode)
        # FrameTimecode
        elif isinstance(timecode, FrameTimecode):
            return timecode.frame_num
        elif timecode is None:
            raise TypeError("Timecode/frame number must be specified!")
        else:
            raise TypeError("Timecode format/type unrecognized.")

    def _parse_timecode_string(self, input: str) -> int:
        """Parses a string based on the three possible forms (in timecode format,
        as an integer number of frames, or floating-point seconds, ending with 's').

        Requires that the `framerate` property is set before calling this method.
        Assuming a framerate of 30.0 FPS, the strings '00:05:00.000', '00:05:00',
        '9000', '300s', and '300.0' are all possible valid values, all representing
        a period of time equal to 5 minutes, 300 seconds, or 9000 frames (at 30 FPS).

        Raises:
            ValueError: Value could not be parsed correctly.
        """
        assert self.framerate is not None
        input = input.strip()
        # Exact number of frames N
        if input.isdigit():
            timecode = int(input)
            if timecode < 0:
                raise ValueError("Timecode frame number must be positive.")
            return timecode
        # Timecode in string format 'HH:MM:SS[.nnn]' or 'MM:SS[.nnn]'
        elif input.find(":") >= 0:
            values = input.split(":")
            # Case of 'HH:MM:SS[.nnn]'
            if len(values) == 3:
                hrs, mins = int(values[0]), int(values[1])
                secs = float(values[2]) if "." in values[2] else int(values[2])
            # Case of 'MM:SS[.nnn]'
            elif len(values) == 2:
                hrs = 0
                mins = int(values[0])
                secs = float(values[1]) if "." in values[1] else int(values[1])
            if not (hrs >= 0 and mins >= 0 and secs >= 0 and mins < 60 and secs < 60):
                raise ValueError(
                    "Invalid timecode range (values outside allowed range)."
                )
            secs += (hrs * 60 * 60) + (mins * 60)
            return self._seconds_to_frames(secs)
        # Try to parse the number as seconds in the format 1234.5 or 1234s
        if input.endswith("s"):
            input = input[:-1]
        if not input.replace(".", "").isdigit():
            raise ValueError(
                "All characters in timecode seconds string must be digits."
            )
        as_float = float(input)
        if as_float < 0.0:
            raise ValueError("Timecode seconds value must be positive.")
        return self._seconds_to_frames(as_float)

    def __iadd__(
        self, other: Union[int, float, str, "FrameTimecode"]
    ) -> "FrameTimecode":
        if isinstance(other, int):
            self.frame_num += other
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                self.frame_num += other.frame_num
            else:
                raise ValueError(
                    "FrameTimecode instances require equal framerate for addition."
                )
        # Check if value to add is in number of seconds.
        elif isinstance(other, float):
            self.frame_num += self._seconds_to_frames(other)
        elif isinstance(other, str):
            self.frame_num += self._parse_timecode_string(other)
        else:
            raise TypeError(
                "Unsupported type for performing addition with FrameTimecode."
            )
        if self.frame_num < 0:  # Required to allow adding negative seconds/frames.
            self.frame_num = 0
        return self

    def __add__(
        self, other: Union[int, float, str, "FrameTimecode"]
    ) -> "FrameTimecode":
        to_return = FrameTimecode(timecode=self)
        to_return += other
        return to_return

    def __isub__(
        self, other: Union[int, float, str, "FrameTimecode"]
    ) -> "FrameTimecode":
        if isinstance(other, int):
            self.frame_num -= other
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                self.frame_num -= other.frame_num
            else:
                raise ValueError(
                    "FrameTimecode instances require equal framerate for subtraction."
                )
        # Check if value to add is in number of seconds.
        elif isinstance(other, float):
            self.frame_num -= self._seconds_to_frames(other)
        elif isinstance(other, str):
            self.frame_num -= self._parse_timecode_string(other)
        else:
            raise TypeError(
                "Unsupported type for performing subtraction with FrameTimecode: %s"
                % type(other)
            )
        if self.frame_num < 0:
            self.frame_num = 0
        return self

    def __sub__(
        self, other: Union[int, float, str, "FrameTimecode"]
    ) -> "FrameTimecode":
        to_return = FrameTimecode(timecode=self)
        to_return -= other
        return to_return

    def __eq__(self, other: Union[int, float, str, "FrameTimecode"]) -> "FrameTimecode":
        if isinstance(other, int):
            return self.frame_num == other
        elif isinstance(other, float):
            return self.get_seconds() == other
        elif isinstance(other, str):
            return self.frame_num == self._parse_timecode_string(other)
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                return self.frame_num == other.frame_num
            else:
                raise TypeError(
                    "FrameTimecode objects must have the same framerate to be compared."
                )
        elif other is None:
            return False
        else:
            raise TypeError(
                "Unsupported type for performing == with FrameTimecode: %s"
                % type(other)
            )

    def __ne__(self, other: Union[int, float, str, "FrameTimecode"]) -> bool:
        return not self == other

    def __lt__(self, other: Union[int, float, str, "FrameTimecode"]) -> bool:
        if isinstance(other, int):
            return self.frame_num < other
        elif isinstance(other, float):
            return self.get_seconds() < other
        elif isinstance(other, str):
            return self.frame_num < self._parse_timecode_string(other)
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                return self.frame_num < other.frame_num
            else:
                raise TypeError(
                    "FrameTimecode objects must have the same framerate to be compared."
                )
        else:
            raise TypeError(
                "Unsupported type for performing < with FrameTimecode: %s" % type(other)
            )

    def __le__(self, other: Union[int, float, str, "FrameTimecode"]) -> bool:
        if isinstance(other, int):
            return self.frame_num <= other
        elif isinstance(other, float):
            return self.get_seconds() <= other
        elif isinstance(other, str):
            return self.frame_num <= self._parse_timecode_string(other)
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                return self.frame_num <= other.frame_num
            else:
                raise TypeError(
                    "FrameTimecode objects must have the same framerate to be compared."
                )
        else:
            raise TypeError(
                "Unsupported type for performing <= with FrameTimecode: %s"
                % type(other)
            )

    def __gt__(self, other: Union[int, float, str, "FrameTimecode"]) -> bool:
        if isinstance(other, int):
            return self.frame_num > other
        elif isinstance(other, float):
            return self.get_seconds() > other
        elif isinstance(other, str):
            return self.frame_num > self._parse_timecode_string(other)
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                return self.frame_num > other.frame_num
            else:
                raise TypeError(
                    "FrameTimecode objects must have the same framerate to be compared."
                )
        else:
            raise TypeError(
                "Unsupported type for performing > with FrameTimecode: %s" % type(other)
            )

    def __ge__(self, other: Union[int, float, str, "FrameTimecode"]) -> bool:
        if isinstance(other, int):
            return self.frame_num >= other
        elif isinstance(other, float):
            return self.get_seconds() >= other
        elif isinstance(other, str):
            return self.frame_num >= self._parse_timecode_string(other)
        elif isinstance(other, FrameTimecode):
            if self.equal_framerate(other.framerate):
                return self.frame_num >= other.frame_num
            else:
                raise TypeError(
                    "FrameTimecode objects must have the same framerate to be compared."
                )
        else:
            raise TypeError(
                "Unsupported type for performing >= with FrameTimecode: %s"
                % type(other)
            )

    # TODO(v1.0): __int__ and __float__ should be removed. Mark as deprecated, and indicate
    # need to use relevant property instead.

    def __int__(self) -> int:
        return self.frame_num

    def __float__(self) -> float:
        return self.get_seconds()

    def __str__(self) -> str:
        return self.get_timecode()

    def __repr__(self) -> str:
        return "%s [frame=%d, fps=%.3f]" % (
            self.get_timecode(),
            self.frame_num,
            self.framerate,
        )

    def __hash__(self) -> int:
        return self.frame_num


COLUMN_NAME_FRAME_NUMBER = "Frame Number"
"""Name of column containing frame numbers in the statsfile CSV."""

COLUMN_NAME_TIMECODE = "Timecode"
"""Name of column containing timecodes in the statsfile CSV."""

##
## StatsManager Exceptions
##


class FrameMetricRegistered(Exception):
    """[DEPRECATED - DO NOT USE] No longer used.

    :meta private:
    """

    pass


class FrameMetricNotRegistered(Exception):
    """[DEPRECATED - DO NOT USE] No longer used.

    :meta private:
    """

    pass


class StatsFileCorrupt(Exception):
    """Raised when frame metrics/stats could not be loaded from a provided CSV file."""

    def __init__(
        self,
        message: str = "Could not load frame metric data data from passed CSV file.",
    ):
        super().__init__(message)


class StatsManager:
    """Provides a key-value store for frame metrics/calculations which can be used
    for two-pass detection algorithms, as well as saving stats to a CSV file.

    Analyzing a statistics CSV file is also very useful for finding the optimal
    algorithm parameters for certain detection methods. Additionally, the data
    may be plotted by a graphing module (e.g. matplotlib) by obtaining the
    metric of interest for a series of frames by iteratively calling get_metrics(),
    after having called the detect_scenes(...) method on the SceneManager object
    which owns the given StatsManager instance.

    Only metrics consisting of `float` or `int` should be used currently.
    """

    def __init__(self, base_timecode: FrameTimecode = None):
        """Initialize a new StatsManager.

        Arguments:
            base_timecode: Timecode associated with this object. Must not be None (default value
                will be removed in a future release).
        """
        # Frame metrics is a dict of frame (int): metric_dict (Dict[str, float])
        # of each frame metric key and the value it represents (usually float).
        self._frame_metrics: Dict[FrameTimecode, Dict[str, float]] = dict()
        self._metric_keys: Set[str] = set()
        self._metrics_updated: bool = (
            False  # Flag indicating if metrics require saving.
        )
        self._base_timecode: Optional[FrameTimecode] = (
            base_timecode  # Used for timing calculations.
        )

    @property
    def metric_keys(self) -> ty.Iterable[str]:
        return self._metric_keys

    def register_metrics(self, metric_keys: Iterable[str]) -> None:
        """Register a list of metric keys that will be used by the detector."""
        self._metric_keys = self._metric_keys.union(set(metric_keys))

    # TODO(v1.0): Change frame_number to a FrameTimecode now that it is just a hash and will
    # be required for VFR support. This API is also really difficult to use, this type should just
    # function like a dictionary.
    def get_metrics(self, frame_number: int, metric_keys: Iterable[str]) -> List[Any]:
        """Return the requested statistics/metrics for a given frame.

        Arguments:
            frame_number (int): Frame number to retrieve metrics for.
            metric_keys (List[str]): A list of metric keys to look up.

        Returns:
            A list containing the requested frame metrics for the given frame number
            in the same order as the input list of metric keys. If a metric could
            not be found, None is returned for that particular metric.
        """
        return [
            self._get_metric(frame_number, metric_key) for metric_key in metric_keys
        ]

    def set_metrics(self, frame_number: int, metric_kv_dict: Dict[str, Any]) -> None:
        """Set Metrics: Sets the provided statistics/metrics for a given frame.

        Arguments:
            frame_number: Frame number to retrieve metrics for.
            metric_kv_dict: A dict mapping metric keys to the
                respective integer/floating-point metric values to set.
        """
        for metric_key in metric_kv_dict:
            self._set_metric(frame_number, metric_key, metric_kv_dict[metric_key])

    def metrics_exist(self, frame_number: int, metric_keys: Iterable[str]) -> bool:
        """Metrics Exist: Checks if the given metrics/stats exist for the given frame.

        Returns:
            bool: True if the given metric keys exist for the frame, False otherwise.
        """
        return all(
            [
                self._metric_exists(frame_number, metric_key)
                for metric_key in metric_keys
            ]
        )

    def is_save_required(self) -> bool:
        """Is Save Required: Checks if the stats have been updated since loading.

        Returns:
            bool: True if there are frame metrics/statistics not yet written to disk,
            False otherwise.
        """
        return self._metrics_updated

    def save_to_csv(
        self,
        csv_file: Union[str, bytes, Path, TextIO],
        base_timecode: Optional[FrameTimecode] = None,
        force_save=True,
    ) -> None:
        """Save To CSV: Saves all frame metrics stored in the StatsManager to a CSV file.

        Arguments:
            csv_file: A file handle opened in write mode (e.g. open('...', 'w')) or a path as str.
            base_timecode: [DEPRECATED] DO NOT USE. For backwards compatibility.
            force_save: If True, writes metrics out even if an update is not required.

        Raises:
            OSError: If `path` cannot be opened or a write failure occurs.
        """
        # TODO(v0.7): Replace with DeprecationWarning that `base_timecode` will be removed in v0.8.
        if base_timecode is not None:
            print("base_timecode is deprecated and has no effect.")

        if not (force_save or self.is_save_required()):
            print("No metrics to write.")
            return

        # If we get a path instead of an open file handle, recursively call ourselves
        # again but with file handle instead of path.
        if isinstance(csv_file, (str, bytes, Path)):
            with open(csv_file, "w") as file:
                self.save_to_csv(csv_file=file, force_save=force_save)
                return

        csv_writer = csv.writer(csv_file, lineterminator="\n")
        metric_keys = sorted(list(self._metric_keys))
        csv_writer.writerow(
            [COLUMN_NAME_FRAME_NUMBER, COLUMN_NAME_TIMECODE] + metric_keys
        )
        frame_keys = sorted(self._frame_metrics.keys())
        print("Writing %d frames to CSV...", len(frame_keys))
        for frame_key in frame_keys:
            frame_timecode = self._base_timecode + frame_key
            csv_writer.writerow(
                [frame_timecode.get_frames() + 1, frame_timecode.get_timecode()]
                + [str(metric) for metric in self.get_metrics(frame_key, metric_keys)]
            )

    @staticmethod
    def valid_header(row: List[str]) -> bool:
        """Check that the given CSV row is a valid header for a statsfile.

        Arguments:
            row: A row decoded from the CSV reader.

        Returns:
            True if `row` is a valid statsfile header, False otherwise.
        """
        if not row or not len(row) >= 2:
            return False
        if row[0] != COLUMN_NAME_FRAME_NUMBER or row[1] != COLUMN_NAME_TIMECODE:
            return False
        return True

    # TODO(v1.0): Create a replacement for a calculation cache that functions like load_from_csv
    # did, but is better integrated with detectors for cached calculations instead of statistics.
    def load_from_csv(self, csv_file: Union[str, bytes, TextIO]) -> Optional[int]:
        """[DEPRECATED] DO NOT USE

        Load all metrics stored in a CSV file into the StatsManager instance. Will be removed in a
        future release after becoming a no-op.

        Arguments:
            csv_file: A file handle opened in read mode (e.g. open('...', 'r')) or a path as str.

        Returns:
            int or None: Number of frames/rows read from the CSV file, or None if the
            input file was blank or could not be found.

        Raises:
            StatsFileCorrupt: Stats file is corrupt and can't be loaded, or wrong file
                was specified.

        :meta private:
        """
        # TODO: Make this an error, then make load_from_csv() a no-op, and finally, remove it.
        print("load_from_csv() is deprecated and will be removed in a future release.")

        # If we get a path instead of an open file handle, check that it exists, and if so,
        # recursively call ourselves again but with file set instead of path.
        if isinstance(csv_file, (str, bytes, Path)):
            if os.path.exists(csv_file):
                with open(csv_file) as file:
                    return self.load_from_csv(csv_file=file)
            # Path doesn't exist.
            return None

        # If we get here, file is a valid file handle in read-only text mode.
        csv_reader = csv.reader(csv_file, lineterminator="\n")
        num_cols = None
        num_metrics = None
        num_frames = None
        # First Row: Frame Num, Timecode, [metrics...]
        try:
            row = next(csv_reader)
            # Backwards compatibility for previous versions of statsfile
            # which included an additional header row.
            if not self.valid_header(row):
                row = next(csv_reader)
        except StopIteration:
            # If the file is blank or we couldn't decode anything, assume the file was empty.
            return None
        if not self.valid_header(row):
            raise StatsFileCorrupt()
        num_cols = len(row)
        num_metrics = num_cols - 2
        if not num_metrics > 0:
            raise StatsFileCorrupt("No metrics defined in CSV file.")
        loaded_metrics = list(row[2:])
        num_frames = 0
        for row in csv_reader:
            metric_dict = {}
            if not len(row) == num_cols:
                raise StatsFileCorrupt(
                    "Wrong number of columns detected in stats file row."
                )
            frame_number = int(row[0])
            # Switch from 1-based to 0-based frame numbers.
            if frame_number > 0:
                frame_number -= 1
            self.set_metrics(frame_number, metric_dict)
            for i, metric in enumerate(row[2:]):
                if metric and metric != "None":
                    try:
                        self._set_metric(frame_number, loaded_metrics[i], float(metric))
                    except ValueError:
                        raise StatsFileCorrupt(
                            "Corrupted value in stats file: %s" % metric
                        ) from ValueError
            num_frames += 1
        self._metric_keys = self._metric_keys.union(set(loaded_metrics))
        print("Loaded %d metrics for %d frames.", num_metrics, num_frames)
        self._metrics_updated = False
        return num_frames

    # TODO: Get rid of these functions and simplify the implementation of this class.

    def _get_metric(self, frame_number: int, metric_key: str) -> Optional[Any]:
        if self._metric_exists(frame_number, metric_key):
            return self._frame_metrics[frame_number][metric_key]
        return None

    def _set_metric(
        self, frame_number: int, metric_key: str, metric_value: Any
    ) -> None:
        self._metrics_updated = True
        if frame_number not in self._frame_metrics:
            self._frame_metrics[frame_number] = dict()
        self._frame_metrics[frame_number][metric_key] = metric_value

    def _metric_exists(self, frame_number: int, metric_key: str) -> bool:
        return (
            frame_number in self._frame_metrics
            and metric_key in self._frame_metrics[frame_number]
        )


class SceneDetector:
    """Base class to inherit from when implementing a scene detection algorithm.

    This API is not yet stable and subject to change.

    This represents a "dense" scene detector, which returns a list of frames where
    the next scene/shot begins in a video.

    Also see the implemented scene detectors in the scenedetect.detectors module
    to get an idea of how a particular detector can be created.
    """

    # TODO(v0.7): Make this a proper abstract base class.

    stats_manager: ty.Optional[StatsManager] = None
    """Optional :class:`StatsManager <scenedetect.stats_manager.StatsManager>` to
    use for caching frame metrics to and from."""

    # TODO(v1.0): Remove - this is a rarely used case for what is now a neglegible performance gain.
    def is_processing_required(self, frame_num: int) -> bool:
        """[DEPRECATED] DO NOT USE

        Test if all calculations for a given frame are already done.

        Returns:
            False if the SceneDetector has assigned _metric_keys, and the
            stats_manager property is set to a valid StatsManager object containing
            the required frame metrics/calculations for the given frame - thus, not
            needing the frame to perform scene detection.

            True otherwise (i.e. the frame_img passed to process_frame is required
            to be passed to process_frame for the given frame_num).
        """
        metric_keys = self.get_metrics()
        return not metric_keys or not (
            self.stats_manager is not None
            and self.stats_manager.metrics_exist(frame_num, metric_keys)
        )

    def stats_manager_required(self) -> bool:
        """Stats Manager Required: Prototype indicating if detector requires stats.

        Returns:
            True if a StatsManager is required for the detector, False otherwise.
        """
        return False

    def get_metrics(self) -> ty.List[str]:
        """Get Metrics:  Get a list of all metric names/keys used by the detector.

        Returns:
            List of strings of frame metric key names that will be used by
            the detector when a StatsManager is passed to process_frame.
        """
        return []

    def process_frame(self, frame_num: int, frame_img: numpy.ndarray) -> ty.List[int]:
        """Process the next frame. `frame_num` is assumed to be sequential.

        Args:
            frame_num (int): Frame number of frame that is being passed. Can start from any value
                but must remain sequential.
            frame_img (numpy.ndarray or None): Video frame corresponding to `frame_img`.

        Returns:
            List[int]: List of frames where scene cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.

        Returns:
            List of frame numbers of cuts to be added to the cutting list.
        """
        return []

    def post_process(self, frame_num: int) -> ty.List[int]:
        """Post Process: Performs any processing after the last frame has been read.

        Prototype method, no actual detection.

        Returns:
            List of frame numbers of cuts to be added to the cutting list.
        """
        return []

    @property
    def event_buffer_length(self) -> int:
        """The amount of frames a given event can be buffered for, in time. Represents maximum
        amount any event can be behind `frame_number` in the result of :meth:`process_frame`.
        """
        return 0


class SparseSceneDetector(SceneDetector):
    """Base class to inherit from when implementing a sparse scene detection algorithm.

    This class will be removed in v1.0 and should not be used.

    Unlike dense detectors, sparse detectors detect "events" and return a *pair* of frames,
    as opposed to just a single cut.

    An example of a SparseSceneDetector is the MotionDetector.
    """

    def process_frame(
        self, frame_num: int, frame_img: numpy.ndarray
    ) -> ty.List[ty.Tuple[int, int]]:
        """Process Frame: Computes/stores metrics and detects any scene changes.

        Prototype method, no actual detection.

        Returns:
            List of frame pairs representing individual scenes
            to be added to the output scene list directly.
        """
        return []

    def post_process(self, frame_num: int) -> ty.List[ty.Tuple[int, int]]:
        """Post Process: Performs any processing after the last frame has been read.

        Prototype method, no actual detection.

        Returns:
            List of frame pairs representing individual scenes
            to be added to the output scene list directly.
        """
        return []


class FlashFilter:
    class Mode(Enum):
        MERGE = 0
        """Merge consecutive cuts shorter than filter length."""
        SUPPRESS = 1
        """Suppress consecutive cuts until the filter length has passed."""

    def __init__(self, mode: Mode, length: int):
        self._mode = mode
        self._filter_length = (
            length  # Number of frames to use for activating the filter.
        )
        self._last_above = None  # Last frame above threshold.
        self._merge_enabled = (
            False  # Used to disable merging until at least one cut was found.
        )
        self._merge_triggered = False  # True when the merge filter is active.
        self._merge_start = None  # Frame number where we started the merge filter.

    @property
    def max_behind(self) -> int:
        """Maximum number of frames a filtered cut can be behind the current frame."""
        return 0 if self._mode == FlashFilter.Mode.SUPPRESS else self._filter_length

    def filter(self, frame_num: int, above_threshold: bool) -> ty.List[int]:
        if not self._filter_length > 0:
            return [frame_num] if above_threshold else []
        if self._last_above is None:
            self._last_above = frame_num
        if self._mode == FlashFilter.Mode.MERGE:
            return self._filter_merge(
                frame_num=frame_num, above_threshold=above_threshold
            )
        elif self._mode == FlashFilter.Mode.SUPPRESS:
            return self._filter_suppress(
                frame_num=frame_num, above_threshold=above_threshold
            )
        raise RuntimeError("Unhandled FlashFilter mode.")

    def _filter_suppress(self, frame_num: int, above_threshold: bool) -> ty.List[int]:
        min_length_met: bool = (frame_num - self._last_above) >= self._filter_length
        if not (above_threshold and min_length_met):
            return []
        # Both length and threshold requirements were satisfied. Emit the cut, and wait until both
        # requirements are met again.
        self._last_above = frame_num
        return [frame_num]

    def _filter_merge(self, frame_num: int, above_threshold: bool) -> ty.List[int]:
        min_length_met: bool = (frame_num - self._last_above) >= self._filter_length
        # Ensure last frame is always advanced to the most recent one that was above the threshold.
        if above_threshold:
            self._last_above = frame_num
        if self._merge_triggered:
            # This frame was under the threshold, see if enough frames passed to disable the filter.
            num_merged_frames = self._last_above - self._merge_start
            if (
                min_length_met
                and not above_threshold
                and num_merged_frames >= self._filter_length
            ):
                self._merge_triggered = False
                return [self._last_above]
            # Keep merging until enough frames pass below the threshold.
            return []
        # Wait for next frame above the threshold.
        if not above_threshold:
            return []
        # If we met the minimum length requirement, no merging is necessary.
        if min_length_met:
            # Only allow the merge filter once the first cut is emitted.
            self._merge_enabled = True
            return [frame_num]
        # Start merging cuts until the length requirement is met.
        if self._merge_enabled:
            self._merge_triggered = True
            self._merge_start = frame_num
        return []


def _mean_pixel_distance(left: numpy.ndarray, right: numpy.ndarray) -> float:
    """Return the mean average distance in pixel values between `left` and `right`.
    Both `left and `right` should be 2 dimensional 8-bit images of the same shape.
    """
    assert len(left.shape) == 2 and len(right.shape) == 2
    assert left.shape == right.shape
    num_pixels: float = float(left.shape[0] * left.shape[1])
    return (
        numpy.sum(numpy.abs(left.astype(numpy.int32) - right.astype(numpy.int32)))
        / num_pixels
    )


def _estimated_kernel_size(frame_width: int, frame_height: int) -> int:
    """Estimate kernel size based on video resolution."""
    # TODO: This equation is based on manual estimation from a few videos.
    # Create a more comprehensive test suite to optimize against.
    size: int = 4 + round(math.sqrt(frame_width * frame_height) / 192)
    if size % 2 == 0:
        size += 1
    return size


class ContentDetector(SceneDetector):
    """Detects fast cuts using changes in colour and intensity between frames.

    The difference is calculated in the HSV color space, and compared against a set threshold to
    determine when a fast cut has occurred.
    """

    # TODO: Come up with some good weights for a new default if there is one that can pass
    # a wider variety of test cases.
    class Components(NamedTuple):
        """Components that make up a frame's score, and their default values."""

        delta_hue: float = 1.0
        """Difference between pixel hue values of adjacent frames."""
        delta_sat: float = 1.0
        """Difference between pixel saturation values of adjacent frames."""
        delta_lum: float = 1.0
        """Difference between pixel luma (brightness) values of adjacent frames."""
        delta_edges: float = 0.0
        """Difference between calculated edges of adjacent frames.

        Edge differences are typically larger than the other components, so the detection
        threshold may need to be adjusted accordingly."""

    DEFAULT_COMPONENT_WEIGHTS = Components()
    """Default component weights. Actual default values are specified in :class:`Components`
    to allow adding new components without breaking existing usage."""

    LUMA_ONLY_WEIGHTS = Components(
        delta_hue=0.0,
        delta_sat=0.0,
        delta_lum=1.0,
        delta_edges=0.0,
    )
    """Component weights to use if `luma_only` is set."""

    FRAME_SCORE_KEY = "content_val"
    """Key in statsfile representing the final frame score after weighed by specified components."""

    METRIC_KEYS = [FRAME_SCORE_KEY, *Components._fields]
    """All statsfile keys this detector produces."""

    @dataclass
    class _FrameData:
        """Data calculated for a given frame."""

        hue: numpy.ndarray
        """Frame hue map [2D 8-bit]."""
        sat: numpy.ndarray
        """Frame saturation map [2D 8-bit]."""
        lum: numpy.ndarray
        """Frame luma/brightness map [2D 8-bit]."""
        edges: Optional[numpy.ndarray]
        """Frame edge map [2D 8-bit, edges are 255, non edges 0]. Affected by `kernel_size`."""

    def __init__(
        self,
        threshold: float = 27.0,
        min_scene_len: int = 15,
        weights: "ContentDetector.Components" = DEFAULT_COMPONENT_WEIGHTS,
        luma_only: bool = False,
        kernel_size: Optional[int] = None,
        filter_mode: FlashFilter.Mode = FlashFilter.Mode.MERGE,
    ):
        """
        Arguments:
            threshold: Threshold the average change in pixel intensity must exceed to trigger a cut.
            min_scene_len: Once a cut is detected, this many frames must pass before a new one can
                be added to the scene list. Can be an int or FrameTimecode type.
            weights: Weight to place on each component when calculating frame score
                (`content_val` in a statsfile, the value `threshold` is compared against).
            luma_only: If True, only considers changes in the luminance channel of the video.
                Equivalent to specifying `weights` as :data:`ContentDetector.LUMA_ONLY`.
                Overrides `weights` if both are set.
            kernel_size: Size of kernel for expanding detected edges. Must be odd integer
                greater than or equal to 3. If None, automatically set using video resolution.
            filter_mode: Mode to use when filtering cuts to meet `min_scene_len`.
        """
        super().__init__()
        self._threshold: float = threshold
        self._min_scene_len: int = min_scene_len
        self._last_above_threshold: Optional[int] = None
        self._last_frame: Optional[ContentDetector._FrameData] = None
        self._weights: ContentDetector.Components = weights
        if luma_only:
            self._weights = ContentDetector.LUMA_ONLY_WEIGHTS
        self._kernel: Optional[numpy.ndarray] = None
        if kernel_size is not None:
            print(kernel_size)
            if kernel_size < 3 or kernel_size % 2 == 0:
                raise ValueError("kernel_size must be odd integer >= 3")
            self._kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)
        self._frame_score: Optional[float] = None
        self._flash_filter = FlashFilter(mode=filter_mode, length=min_scene_len)

    def get_metrics(self):
        return ContentDetector.METRIC_KEYS

    def is_processing_required(self, frame_num):
        return True

    def _calculate_frame_score(self, frame_num: int, frame_img: numpy.ndarray) -> float:
        """Calculate score representing relative amount of motion in `frame_img` compared to
        the last time the function was called (returns 0.0 on the first call)."""
        # TODO: Add option to enable motion estimation before calculating score components.
        # TODO: Investigate methods of performing cheaper alternatives, e.g. shifting or resizing
        # the frame to simulate camera movement, using optical flow, etc...

        # Convert image into HSV colorspace.
        hue, sat, lum = cv2.split(cv2.cvtColor(frame_img, cv2.COLOR_BGR2HSV))

        # Performance: Only calculate edges if we have to.
        calculate_edges: bool = (
            self._weights.delta_edges > 0.0
        ) or self.stats_manager is not None
        edges = self._detect_edges(lum) if calculate_edges else None

        if self._last_frame is None:
            # Need another frame to compare with for score calculation.
            self._last_frame = ContentDetector._FrameData(hue, sat, lum, edges)
            return 0.0

        score_components = ContentDetector.Components(
            delta_hue=_mean_pixel_distance(hue, self._last_frame.hue),
            delta_sat=_mean_pixel_distance(sat, self._last_frame.sat),
            delta_lum=_mean_pixel_distance(lum, self._last_frame.lum),
            delta_edges=(
                0.0
                if edges is None
                else _mean_pixel_distance(edges, self._last_frame.edges)
            ),
        )

        frame_score: float = sum(
            component * weight
            for (component, weight) in zip(score_components, self._weights)
        ) / sum(abs(weight) for weight in self._weights)

        # Record components and frame score if needed for analysis.
        if self.stats_manager is not None:
            metrics = {self.FRAME_SCORE_KEY: frame_score}
            metrics.update(score_components._asdict())
            self.stats_manager.set_metrics(frame_num, metrics)

        # Store all data required to calculate the next frame's score.
        self._last_frame = ContentDetector._FrameData(hue, sat, lum, edges)
        return frame_score

    def process_frame(self, frame_num: int, frame_img: numpy.ndarray) -> List[int]:
        """Process the next frame. `frame_num` is assumed to be sequential.

        Args:
            frame_num (int): Frame number of frame that is being passed. Can start from any value
                but must remain sequential.
            frame_img (numpy.ndarray or None): Video frame corresponding to `frame_img`.

        Returns:
            List[int]: List of frames where scene cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """
        self._frame_score = self._calculate_frame_score(frame_num, frame_img)
        if self._frame_score is None:
            return []

        above_threshold: bool = self._frame_score >= self._threshold
        return self._flash_filter.filter(
            frame_num=frame_num, above_threshold=above_threshold
        )

    def _detect_edges(self, lum: numpy.ndarray) -> numpy.ndarray:
        """Detect edges using the luma channel of a frame.

        Arguments:
            lum: 2D 8-bit image representing the luma channel of a frame.

        Returns:
            2D 8-bit image of the same size as the input, where pixels with values of 255
            represent edges, and all other pixels are 0.
        """
        # Initialize kernel.
        if self._kernel is None:
            kernel_size = _estimated_kernel_size(lum.shape[1], lum.shape[0])
            self._kernel = numpy.ones((kernel_size, kernel_size), numpy.uint8)

        # Estimate levels for thresholding.
        # TODO: Add config file entries for sigma, aperture/kernel size, etc.
        sigma: float = 1.0 / 3.0
        median = numpy.median(lum)
        low = int(max(0, (1.0 - sigma) * median))
        high = int(min(255, (1.0 + sigma) * median))

        # Calculate edges using Canny algorithm, and reduce noise by dilating the edges.
        # This increases edge overlap leading to improved robustness against noise and slow
        # camera movement. Note that very large kernel sizes can negatively affect accuracy.
        edges = cv2.Canny(lum, low, high)
        return cv2.dilate(edges, self._kernel)

    @property
    def event_buffer_length(self) -> int:
        return self._flash_filter.max_behind


class AdaptiveDetector(ContentDetector):
    """Two-pass detector that calculates frame scores with ContentDetector, and then applies
    a rolling average when processing the result that can help mitigate false detections
    in situations such as camera movement.
    """

    ADAPTIVE_RATIO_KEY_TEMPLATE = "adaptive_ratio{luma_only} (w={window_width})"

    def __init__(
        self,
        adaptive_threshold: float = 3.0,
        min_scene_len: int = 15,
        window_width: int = 2,
        min_content_val: float = 15.0,
        weights: ContentDetector.Components = ContentDetector.DEFAULT_COMPONENT_WEIGHTS,
        luma_only: bool = False,
        kernel_size: Optional[int] = None,
        video_manager=None,
        min_delta_hsv: Optional[float] = None,
    ):
        """
        Arguments:
            adaptive_threshold: Threshold (float) that score ratio must exceed to trigger a
                new scene (see frame metric adaptive_ratio in stats file).
            min_scene_len: Once a cut is detected, this many frames must pass before a new one can
                be added to the scene list. Can be an int or FrameTimecode type.
            window_width: Size of window (number of frames) before and after each frame to
                average together in order to detect deviations from the mean. Must be at least 1.
            min_content_val: Minimum threshold (float) that the content_val must exceed in order to
                register as a new scene. This is calculated the same way that `detect-content`
                calculates frame score based on `weights`/`luma_only`/`kernel_size`.
            weights: Weight to place on each component when calculating frame score
                (`content_val` in a statsfile, the value `threshold` is compared against).
                If omitted, the default ContentDetector weights are used.
            luma_only: If True, only considers changes in the luminance channel of the video.
                Equivalent to specifying `weights` as :data:`ContentDetector.LUMA_ONLY`.
                Overrides `weights` if both are set.
            kernel_size: Size of kernel to use for post edge detection filtering. If None,
                automatically set based on video resolution.
            video_manager: [DEPRECATED] DO NOT USE. For backwards compatibility only.
            min_delta_hsv: [DEPRECATED] DO NOT USE. Use `min_content_val` instead.
        """
        # TODO(v0.7): Replace with DeprecationWarning that `video_manager` and `min_delta_hsv` will
        # be removed in v0.8.
        if video_manager is not None:
            print("video_manager is deprecated, use video instead.")
        if min_delta_hsv is not None:
            print("min_delta_hsv is deprecated, use min_content_val instead.")
            min_content_val = min_delta_hsv
        if window_width < 1:
            raise ValueError("window_width must be at least 1.")

        super().__init__(
            threshold=255.0,
            min_scene_len=0,
            weights=weights,
            luma_only=luma_only,
            kernel_size=kernel_size,
        )

        # TODO: Turn these options into properties.
        self.min_scene_len = min_scene_len
        self.adaptive_threshold = adaptive_threshold
        self.min_content_val = min_content_val
        self.window_width = window_width

        self._adaptive_ratio_key = AdaptiveDetector.ADAPTIVE_RATIO_KEY_TEMPLATE.format(
            window_width=window_width, luma_only="" if not luma_only else "_lum"
        )
        self._first_frame_num = None

        # NOTE: This must be different than `self._last_scene_cut` which is used by the base class.
        self._last_cut: Optional[int] = None

        self._buffer = []

    @property
    def event_buffer_length(self) -> int:
        """Number of frames any detected cuts will be behind the current frame due to buffering."""
        return self.window_width

    def get_metrics(self) -> List[str]:
        """Combines base ContentDetector metric keys with the AdaptiveDetector one."""
        return super().get_metrics() + [self._adaptive_ratio_key]

    def stats_manager_required(self) -> bool:
        """Not required for AdaptiveDetector."""
        return False

    def process_frame(
        self, frame_num: int, frame_img: Optional[np.ndarray]
    ) -> List[int]:
        """Process the next frame. `frame_num` is assumed to be sequential.

        Args:
            frame_num (int): Frame number of frame that is being passed. Can start from any value
                but must remain sequential.
            frame_img (numpy.ndarray or None): Video frame corresponding to `frame_img`.

        Returns:
            List[int]: List of frames where scene cuts have been detected. There may be 0
            or more frames in the list, and not necessarily the same as frame_num.
        """

        # TODO(#283): Merge this with ContentDetector and turn it on by default.

        super().process_frame(frame_num=frame_num, frame_img=frame_img)

        # Initialize last scene cut point at the beginning of the frames of interest.
        if self._last_cut is None:
            self._last_cut = frame_num

        required_frames = 1 + (2 * self.window_width)
        self._buffer.append((frame_num, self._frame_score))
        if not len(self._buffer) >= required_frames:
            return []
        self._buffer = self._buffer[-required_frames:]
        (target_frame, target_score) = self._buffer[self.window_width]
        average_window_score = sum(
            score
            for i, (_frame, score) in enumerate(self._buffer)
            if i != self.window_width
        ) / (2.0 * self.window_width)

        average_is_zero = abs(average_window_score) < 0.00001

        adaptive_ratio = 0.0
        if not average_is_zero:
            adaptive_ratio = min(target_score / average_window_score, 255.0)
        elif average_is_zero and target_score >= self.min_content_val:
            # if we would have divided by zero, set adaptive_ratio to the max (255.0)
            adaptive_ratio = 255.0
        if self.stats_manager is not None:
            self.stats_manager.set_metrics(
                target_frame, {self._adaptive_ratio_key: adaptive_ratio}
            )

        # Check to see if adaptive_ratio exceeds the adaptive_threshold as well as there
        # being a large enough content_val to trigger a cut
        threshold_met: bool = (
            adaptive_ratio >= self.adaptive_threshold
            and target_score >= self.min_content_val
        )
        min_length_met: bool = (frame_num - self._last_cut) >= self.min_scene_len
        if threshold_met and min_length_met:
            self._last_cut = target_frame
            return [target_frame]
        return []

    def get_content_val(self, frame_num: int) -> Optional[float]:
        """Returns the average content change for a frame."""
        # TODO(v0.7): Add DeprecationWarning that `get_content_val` will be removed in v0.7.
        print(
            "get_content_val is deprecated and will be removed. Lookup the value"
            " using a StatsManager with ContentDetector.FRAME_SCORE_KEY."
        )
        if self.stats_manager is not None:
            return self.stats_manager.get_metrics(
                frame_num, [ContentDetector.FRAME_SCORE_KEY]
            )[0]
        return 0.0

    def post_process(self, _unused_frame_num: int):
        """Not required for AdaptiveDetector."""
        return []
