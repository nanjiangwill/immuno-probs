# Pygor is part of the IGoR (Inference and Generation of Repertoires)
# software. This Python package can be used for post processing of IGoR
# generated output files.
# Copyright (C) 2018 Quentin Marcou & Wout van Helvoirt

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


"""Contains conversion tools used in pygor."""


def nucleotides_to_integers(seq):
    """Converts a nucleotide sequence to an interger representation.

    The base characters in the nucleotide string (A, C, G and T) are converted
    to the following: A -> 0, C -> 1, G -> 2 and T -> 3. The combined uppercase
    string is returned.

    Parameters
    ----------
    seq : string
        A nucleotide sequence string.

    Returns
    -------
    string
        The interger representation string for the given nucleotide sequence.

    """
    int_sequence = []
    for i in seq.upper():
        if i == 'A':
            int_sequence.append(0)
        elif i == 'C':
            int_sequence.append(1)
        elif i == 'G':
            int_sequence.append(2)
        elif i == 'T':
            int_sequence.append(3)
    return ''.join(int_sequence)


def integers_to_nucleotides(seq):
    """Converts a integer sequence to an nucleotide representation.

    The base characters in the integer string (0, 1, 2 and 3) are converted
    to the following: 0 -> A, 1 -> C, 2 -> G and 3 -> T. The combined string
    is returned.

    Parameters
    ----------
    seq : string
        A integer sequence string.

    Returns
    -------
    string
        The nucleotide representation string for the given integer sequence.

    """
    nuc_sequence = []
    for i in seq:
        if int(i) == 0:
            nuc_sequence.append('A')
        elif int(i) == 1:
            nuc_sequence.append('C')
        elif int(i) == 2:
            nuc_sequence.append('G')
        elif int(i) == 3:
            nuc_sequence.append('T')
    return ''.join(nuc_sequence)


def get_reverse_complement(seq):
    """Converts a nucleotide sequence to reverse complement.

    The base characters in the nucleotide string (A, C, G and T) are converted
    to the following: A <-> T and C <-> G. The combined uppercase string is
    returned.

    Parameters
    ----------
    seq : string
        A nucleotide sequence string.

    Returns
    -------
    string
        The reverse complemented nucleotide sequence.

    """
    reverse_complement_seq = []
    for i in seq.upper():
        if i == 'A':
            reverse_complement_seq.append('T')
        elif i == 'C':
            reverse_complement_seq.append('G')
        elif i == 'G':
            reverse_complement_seq.append('C')
        elif i == 'T':
            reverse_complement_seq.append('A')
    return ''.join(reverse_complement_seq)


def type_string_to_list(in_str, dtype=float, l_bound='(', r_bound=')', sep=','):
    """Converts a string representation of an array to a python list.

    Removes the given boundary characters from the string and separates the
    individual items on the given seperator character. Each item is converted to
    the given dtype. The python list is returned.

    Parameters
    ----------
    in_str : string
        A array representated as string.
    dtype : type
        The dtype to used for converting the individual the list elements. By
        default uses float.
    l_bound : string
        A string specifying the left boundary character(s).
    r_bound : string
        A string specifying the right boundary character(s).
    sep : string
        The separator character used in the input string.

    Returns
    -------
    list
        The converted input string as python list.

    """
    if len(in_str) > (len(l_bound) + len(r_bound)):

        # Check if start and end of the string match the boundary characters.
        if in_str.find(l_bound) != 0:
            print('Beginning character not found')
        elif in_str.find(r_bound) != (
                len(str) - len(r_bound)):
            print('Ending character not found')

        # Strip the boundary characters, split on seperator and small cleanup.
        converted_str = [dtype(i.strip(' \"\''))
                         for i in in_str.lstrip(l_bound).rstrip(r_bound).split(sep)]
    return converted_str
