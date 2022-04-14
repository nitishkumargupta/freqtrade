import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
from pandas import DataFrame, read_csv, to_datetime

from freqtrade import misc
from freqtrade.configuration import TimeRange
from freqtrade.constants import DEFAULT_DATAFRAME_COLUMNS, ListPairsWithTimeframes, TradeList
from freqtrade.data.converter import trades_dict_to_list

from .idatahandler import IDataHandler


logger = logging.getLogger(__name__)


class CSVDataHandler(IDataHandler):

    _use_zip = False
    _columns = DEFAULT_DATAFRAME_COLUMNS

    @classmethod
    def ohlcv_get_available_data(cls, datadir: Path) -> ListPairsWithTimeframes:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        :param datadir: Directory to search for ohlcv files
        :return: List of Tuples of (pair, timeframe)
        """
        _tmp = [re.search(r'^([a-zA-Z_]+)\-(\d+\S+)(?=.csv)', p.name)
                for p in datadir.glob(f"*.{cls._get_file_extension()}")]
        return [(match[1].replace('_', '/'), match[2]) for match in _tmp
                if match and len(match.groups()) > 1]

    @classmethod
    def ohlcv_get_pairs(cls, datadir: Path, timeframe: str) -> List[str]:
        """
        Returns a list of all pairs with ohlcv data available in this datadir
        for the specified timeframe
        :param datadir: Directory to search for ohlcv files
        :param timeframe: Timeframe to search pairs for
        :return: List of Pairs
        """

        _tmp = [re.search(r'^(\S+)(?=\-' + timeframe + '.csv)', p.name)
                for p in datadir.glob(f"*{timeframe}.{cls._get_file_extension()}")]
        # Check if regex found something and only return these results
        return [match[0].replace('_', '/') for match in _tmp if match]

    def ohlcv_store(self, pair: str, timeframe: str, data: DataFrame) -> None:
        """
        Store data in csv format "values".
            format looks as follows:
            [[<date>,<open>,<high>,<low>,<close>,<volume>]]
        :param pair: Pair - used to generate filename
        :param timeframe: Timeframe - used to generate filename
        :param data: Dataframe containing OHLCV data
        :return: None
        """
        filename = self._pair_to_filename_without_underscore(self._datadir, pair, timeframe)
        _data = data.copy()
        # Convert date to int
        parsed_date = to_datetime(_data['date'])
        date = parsed_date.dt.strftime('%Y%m%d')
        # _data['date'] = _data['date'].view(np.int64) // 1000 // 1000 # Default date
        _data['date'] = date

        # Reset index, select only appropriate columns and save as json
        _data.loc[:, self._columns].to_csv(
            filename, header=False,index=False,float_format='%g')

    def _ohlcv_load(self, pair: str, timeframe: str,
                    timerange: Optional[TimeRange] = None,
                    ) -> DataFrame:
        """
        Internal method used to load data for one pair from disk.
        Implements the loading and conversion to a Pandas dataframe.
        Timerange trimming and dataframe validation happens outside of this method.
        :param pair: Pair to load data
        :param timeframe: Timeframe (e.g. "5m")
        :param timerange: Limit data to be loaded to this timerange.
                        Optionally implemented by subclasses to avoid loading
                        all data where possible.
        :return: DataFrame with ohlcv data, or empty DataFrame
        """
        filename = self._pair_to_filename_without_underscore(self._datadir, pair, timeframe)
        if not filename.exists():
            return DataFrame(columns=self._columns)
        try:
            pairdata = read_csv(filename, header=None)
            pairdata.columns = self._columns
        except ValueError:
            logger.error(f"Could not load data for {pair}.")
            return DataFrame(columns=self._columns)
        pairdata = pairdata.astype(dtype={'open': 'float', 'high': 'float',
                                          'low': 'float', 'close': 'float', 'volume': 'float'})
        pairdata['date'] = to_datetime(pairdata['date'],
                                       unit='ms',
                                       utc=True,
                                       infer_datetime_format=True)
        return pairdata

    def ohlcv_purge(self, pair: str, timeframe: str) -> bool:
        """
        Remove data for this pair
        :param pair: Delete data for this pair.
        :param timeframe: Timeframe (e.g. "5m")
        :return: True when deleted, false if file did not exist.
        """
        filename = self._pair_to_filename_without_underscore(self._datadir, pair, timeframe)
        if filename.exists():
            filename.unlink()
            return True
        return False

    def ohlcv_append(self, pair: str, timeframe: str, data: DataFrame) -> None:
        """
        Append data to existing data structures
        :param pair: Pair
        :param timeframe: Timeframe this ohlcv data is for
        :param data: Data to append.
        """
        raise NotImplementedError()

    @classmethod
    def trades_get_pairs(cls, datadir: Path) -> List[str]:
        """
        Returns a list of all pairs for which trade data is available in this
        :param datadir: Directory to search for ohlcv files
        :return: List of Pairs
        """
        _tmp = [re.search(r'^(\S+)(?=\-trades.csv)', p.name)
                for p in datadir.glob(f"*trades.{cls._get_file_extension()}")]
        # Check if regex found something and only return these results to avoid exceptions.
        return [match[0].replace('_', '/') for match in _tmp if match]

    def trades_store(self, pair: str, data: TradeList) -> None:
        """
        Store trades data (list of Dicts) to file
        :param pair: Pair - used for filename
        :param data: List of Lists containing trade data,
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        filename = self._pair_trades_filename(self._datadir, pair)
        DataFrame(data, columns=DEFAULT_TRADES_COLUMNS).to_csv(
            filename)

    def trades_append(self, pair: str, data: TradeList):
        """
        Append data to existing files
        :param pair: Pair - used for filename
        :param data: List of Lists containing trade data,
                     column sequence as in DEFAULT_TRADES_COLUMNS
        """
        raise NotImplementedError()

    def _trades_load(self, pair: str, timerange: Optional[TimeRange] = None) -> TradeList:
        """
        Load a pair from file, either .json.gz or .json
        # TODO: respect timerange ...
        :param pair: Load trades for this pair
        :param timerange: Timerange to load trades for - currently not implemented
        :return: List of trades
        """
        filename = self._pair_trades_filename(self._datadir, pair)
        tradesdata: DataFrame = read_csv(filename)

        if not tradesdata:
            return []

        if isinstance(tradesdata[0], dict):
            # Convert trades dict to list
            logger.info("Old trades format detected - converting")
            tradesdata = trades_dict_to_list(tradesdata)
            pass
        return tradesdata

    @classmethod
    def _get_file_extension(cls):
        return "csv"
