# Create IGoR models and calculate the generation probability of V(D)J and
# CDR3 sequences. Copyright (C) 2019 Wout van Helvoirt

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""Contains I/O related functions used in immuno_probs."""


import os
from shutil import copy2

from Bio import SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser
import pandas


def create_directory_path(directory):
    """Updates and creates given directory path by adding a number at the end.

    Parameters
    ----------
    directory : string
        A directory path location to create recursively.

    Returns
    -------
    str
        The (updated) output directory location path.

    """
    # Check if the directory name is unique, modify name if necessary.
    dir_count = 1
    updated_directory = directory

    # Keep modifying the name until it doesn't exist.
    while os.path.isdir(os.path.join(directory, updated_directory)):
        updated_directory = str(directory) + '_' + str(dir_count)
        dir_count += 1

    # Finally create directory's recursively if not exists.
    if not os.path.isdir(updated_directory):
        os.makedirs(updated_directory)

    return updated_directory


def is_fasta(file):
    """Checks if the input file is valid fasta.

    Parameters
    ----------
    file : string
        Location of the FASTA file to be tested.

    """
    with open(file, "r") as fasta_file:
        return any(SeqIO.parse(fasta_file, "fasta"))


def is_separated(file, separator):
    """Checks if the input file is a valid separated file.

    Parameters
    ----------
    file : string
        Location of the separated file to be tested.
    separator : string
        A separator character used for separating the fields in the file.

    """
    dataframe = pandas.read_csv(file, sep=separator, comment='#', header=0, nrows=100)
    return not dataframe.empty


def read_fasta_as_dataframe(file, col):
    """Creates a pandas.DataFrame from the FASTA file.

    The dataframe contains header name and sequence columns containing the
    corresponding FASTA data.

    Parameters
    ----------
    file : string
        Location of the FASTA file to be read in.
    col : string
        The name of the FASTA sequence column.

    """
    # Create a dataframe and read in the fasta file.
    fasta_df = pandas.DataFrame(columns=[col])
    with open(file, 'r') as fasta_file:
        for _, sequence in SimpleFastaParser(fasta_file):
            fasta_df = fasta_df.append({
                col: sequence.upper()
            }, ignore_index=True)
    return fasta_df


def read_separated_to_dataframe(file, separator, index_col=None, cols=None):
    """Read in a separated file as pandas.DataFrame.

    Parameters
    ----------
    file : string
        File path to be read in as dataframe.
    separator : string
        A separator character used for separating the fields in the file.
    index_col : str, optional
        The name of the index column to use. If specified and given column is
        not found in the dataframe, the index values are generated. (default: No
        index column)
    cols : list, optional
        Containing column names to keep in the output file. The order will
        change the output file column formatting (default: includes all
        columns in the output file).

    Notes
    -----
        This function uses the global SEPARATOR variable to set the separator
        string for the input file. Comments ('#') in the file are skipped.
        If the given index column contains NA values, the column is ignored.

    Raises
    ------
    KeyError
        If a given column is not found in the input data file.

    """
    # Read in columns of the given file.
    if cols:
        if index_col:
            cols.insert(0, index_col)
        separated_df = pandas.read_csv(file, sep=separator, comment='#', header=0,
                                       usecols=lambda value: value in cols)
    else:
        separated_df = pandas.read_csv(file, sep=separator, comment='#', header=0)

    # Set the index column, only use if no NA values.
    if index_col and index_col in separated_df.columns:
        if not separated_df[index_col].isna().any():
            separated_df.set_index(index_col, inplace=True)

    return separated_df


def write_dataframe_to_separated(dataframe, filename, directory, separator, index_name=None):
    """Writes a pandas.DataFrame to a separated formatted file.

    If the file already exists, a number will be appended to the filename.
    The given output directory is created recursively if it does not exist.
    The column names in the dataframe is used as first line in the file.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The dataframe to be written to the separated file.
    filename : string
        Base filename for writting the file, excluding the extension.
    directory : string
        A directory path location to create recursively.
    separator : string
        A separator character used for separating the fields in the file.
    index_name : string, optional
        The output column name for the dataframe index (default: will not write
        the index to the file).

    Returns
    -------
    tuple
        Containing the output directory and the name of the file that has been
        written to disk.

    """
    # Check if the filename is unique, modify name if necessary.
    file_count = 1
    updated_filename = filename

    extension = '.csv'
    if separator == '\t':
        extension = '.tsv'

    # Keep modifying the filename until it doesn't exist.
    while os.path.isfile(os.path.join(directory, updated_filename + extension)):
        updated_filename = str(filename) + '_' + str(file_count)
        file_count += 1

    # Write dataframe contents to separated file and return info.
    enable_index = False
    if index_name:
        enable_index = True
    pandas.DataFrame.to_csv(
        dataframe, path_or_buf=os.path.join(directory, updated_filename + extension),
        sep=separator, index=enable_index, index_label=index_name, na_rep='NA')
    return (directory, updated_filename + extension)


def preprocess_separated_file(directory, file, in_sep, out_sep, index_col=None, cols=None):
    """Function for formatting the input sequence file for IGoR.

    Parameters
    ----------
    directory : str
        A directory path to write the file to.
    file : str
        A separated data file path to process for IGoR.
    in_sep : str
        The input file seperator.
    out_sep : str
        The wanted output file seperator.
    index_col : str, optional
        The name of the index column to use. If specified and given column is
        not found in the dataframe, the index values are generated. (default: No
        index column)
    cols : list, optional
        Containing column names to keep in the output file. The order will
        change the output file column formatting (default: includes all
        columns in the output file).

    Returns
    -------
    str
        A string file path to the newly created file.

    Notes
    -----
        Returns the input file path if no changes will be applied to the file.
        This means, the input seperator and output seperator are equal and the
        columns attribute has not been specified.

    Raises
    ------
    KeyError
        If a given column is not found in the input data file.

    """
    # If the seperators are the same and no columns are given, return the input.
    if out_sep == in_sep and cols is None:
        return file

    # Create the output directory.
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Open the sequence input file.
    sequence_df = read_separated_to_dataframe(
        file=file,
        separator=in_sep,
        index_col=index_col,
        cols=cols)

    # Write the new pandas dataframe to a separated file.
    directory, filename = write_dataframe_to_separated(
        dataframe=sequence_df,
        filename=os.path.basename(str(file)),
        directory=directory,
        separator=out_sep,
        index_name=index_col)
    return os.path.join(directory, filename)


def preprocess_reference_file(directory, file, index):
    """Function for formatting the IMGT reference genome files for IGoR.

    Parameters
    ----------
    directory : str
        A directory path to write the file to.
    file : str
        A FASTA file path for a reference genomic template file.
    index : int
        Index of the header line to keep after splitting on '|'.

    Returns
    -------
    str
        A string file path to the new reference FASTA file.

    """
    # Create the output directory.
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Open the fasta file and update the fasta header.
    records = list(SeqIO.parse(file, "fasta"))
    for rec in records:
        rec.id = rec.description.split('|')[index]
        rec.description = ""

    # Write out the modified file.
    updated_path = os.path.join(directory, os.path.basename(file))
    SeqIO.write(records, updated_path, "fasta")
    return updated_path


def copy_to_dir(directory, file, extension):
    """Function for copying file to directory and modifying the extension.

    Parameters
    ----------
    directory : str
        A directory path to write the file to.
    file : str
        A FASTA file path for a reference genomic template file.
    extension : str
        An extension name, if the same, file will NOT be coppied.

    Returns
    -------
    str
        A string file path to the new file location and name.

    """
    # Check if file name extension if different, or return.
    filename, file_extension = os.path.splitext(os.path.basename(file))
    if file_extension == str('.' + extension):
        return file

    # Copy file to given directory if necessary.
    output_file = os.path.join(directory, filename + '.' + extension)
    copy2(file, output_file)
    return output_file
