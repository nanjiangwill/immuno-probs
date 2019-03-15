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


"""OlgaContainer class for generating and evaluating CDR3 sequences with OLGA."""


import olga.sequence_generation as olga_seq_gen
import olga.generation_probability as olga_pgen
import pandas

from immuno_probs.util.constant import get_num_threads
from immuno_probs.util.conversion import nucleotides_to_aminoacids
from immuno_probs.util.exception import OlgaException
from immuno_probs.util.processing import multiprocess_array


class OlgaContainer(object):
    """Class generating and evaluating CDR3 sequences using an IGoR model.

    Parameters
    ----------
    igor_model : IgorLoader object
        IgorLoader object containing the loaded IGoR VJ or VDJ model.

    Methods
    -------
    generate(num_seqs)
        Returns pandas.DataFrame with NT and AA CDR3 sequences.
    evaluate(seq)
        Returns the generation probability value for the given sequence.

    """
    def __init__(self, igor_model):
        super(OlgaContainer, self).__init__()
        self.igor_model = igor_model

    def generate(self, num_seqs):
        """Generate a given number of CDR3 sequences through OLGA.

        Parameters
        ----------
        num_seqs : int
            An integer specifying the number of sequences to generate.

        Returns
        -------
        pandas.DataFrame
            Containing columns with sequence index - 'seq_index', nucleotide
            CDR3 sequence - 'nt_sequence', ammino acid CDR3 sequence -
            'aa_sequence', the index of the chosen V gene - 'gene_choice_v'
            and the index of the chosen J gene - 'gene_choice_j'.

        """
        # Create the dataframe and set the generation objects.
        generated_seqs = pandas.DataFrame(
            columns=['seq_index', 'nt_sequence', 'aa_sequence', 'gene_choice_v',
                     'gene_choice_j'])
        seq_gen_model = None
        if self.igor_model.get_type() == "VDJ":
            seq_gen_model = olga_seq_gen.SequenceGenerationVDJ(
                self.igor_model.get_generative_model(),
                self.igor_model.get_genomic_data())
        elif self.igor_model.get_type() == "VJ":
            seq_gen_model = olga_seq_gen.SequenceGenerationVJ(
                self.igor_model.get_generative_model(),
                self.igor_model.get_genomic_data())
        else:
            raise OlgaException("OLGA could not create a SequenceGeneration object")

        # Collect the gene names from the genomic data model.
        v_gene_names = [V[0] for V in self.igor_model.get_genomic_data().genV]
        j_gene_names = [J[0] for J in self.igor_model.get_genomic_data().genJ]

        # Generate the sequences, add them to the dataframe and return.
        for i in range(num_seqs):
            generated_seq = seq_gen_model.gen_rnd_prod_CDR3()
            generated_seqs = generated_seqs.append({
                'seq_index': i,
                'nt_sequence': generated_seq[0],
                'aa_sequence': generated_seq[1],
                'gene_choice_v': v_gene_names[generated_seq[2]],
                'gene_choice_j': j_gene_names[generated_seq[3]],
            }, ignore_index=True)
        return generated_seqs

    @staticmethod
    def _evaluate(args):
        """Evaluate a given nucleotide CDR3 sequence through OLGA.

        Parameters
        ----------
        args : list
            The arguments from the multiprocess_array function. Consists of an
            pandas.DataFrame and additional kwargs like the
            GenerationProbability object, the column name containing the
            nucleotide sequences and a boolean for usingf V/J masks.

        Returns
        -------
        pandas.DataFrame
            Containing columns sequence index number - 'seq_index', the
            generation probability of nucleotide sequence if
            given - 'nt_pgen_estimate' and the generation probability of
            aminoacid sequence if given - 'aa_pgen_estimate'.

        """
        # Set the arguments and pandas.DataFrame.
        ary, kwargs = args
        model = kwargs["model"]
        pgen_seqs = pandas.DataFrame(ary['seq_index'], columns=['seq_index'])

        # Evaluate the sequences, add them to the dataframe and return.
        if set(['gene_choice_v', 'gene_choice_j']).issubset(ary.columns):
            for i, row in ary.iterrows():
                if 'nt_sequence' in ary.columns:
                    pgen_seqs.loc[i, 'nt_pgen_estimate'] = model.compute_nt_CDR3_pgen(
                        row['nt_sequence'], row['gene_choice_v'], row['gene_choice_j'])
                if 'aa_sequence' in ary.columns:
                    pgen_seqs.loc[i, 'aa_pgen_estimate'] = model.compute_aa_CDR3_pgen(
                        row['aa_sequence'], row['gene_choice_v'], row['gene_choice_j'])
        else:
            for i, row in ary.iterrows():
                if 'nt_sequence' in ary.columns:
                    pgen_seqs.loc[i, 'nt_pgen_estimate'] = model.compute_nt_CDR3_pgen(
                        row['nt_sequence'])
                if 'aa_sequence' in ary.columns:
                    pgen_seqs.loc[i, 'aa_pgen_estimate'] = model.compute_aa_CDR3_pgen(
                        row['aa_sequence'])
        return pgen_seqs

    def evaluate(self, seqs):
        """Evaluate a given nucleotide CDR3 sequence through OLGA.

        Parameters
        ----------
        seqs : pandas.DataFrame
            A pandas dataframe object containing column with nucleotide CDR3
            sequences - 'nt_sequence' and/or amino acid sequences - 'aa_sequence'.

        Returns
        -------
        pandas.DataFrame
            Containing columns sequence index number - 'seq_index', the
            generation probability of nucleotide sequence if
            given - 'nt_pgen_estimate' and the generation probability of
            aminoacid sequence if given - 'aa_pgen_estimate'.

        Notes
        -----
        This fucntion also checks if the given input sequence file contains the
        'gene_choice_v' and 'gene_choice_j' columns. If so, then the V and J
        gene masks in these columns are used to incease accuracy of the
        generation probabality.

        """
        # Set the evaluation objects.
        pgen_model = None
        if self.igor_model.get_type() == "VDJ":
            pgen_model = olga_pgen.GenerationProbabilityVDJ(
                self.igor_model.get_generative_model(),
                self.igor_model.get_genomic_data())
        elif self.igor_model.get_type() == "VJ":
            pgen_model = olga_pgen.GenerationProbabilityVJ(
                self.igor_model.get_generative_model(),
                self.igor_model.get_genomic_data())
        else:
            raise OlgaException("OLGA could not create a GenerationProbability object")

        # Insert amino acid sequence column if not existent.
        if 'nt_sequence' in seqs.columns and not 'aa_sequence' in seqs.columns:
            seqs.insert(
                seqs.columns.get_loc('nt_sequence') + 1,
                'aa_sequence', '')
            for i, row in seqs.iterrows():
                seqs.loc[i, 'aa_sequence'] = nucleotides_to_aminoacids(
                    row['nt_sequence'])

        # Use multiprocessing to evaluate the sequences in chunks and return.
        result = multiprocess_array(ary=seqs,
                                    func=self._evaluate,
                                    num_workers=get_num_threads(),
                                    model=pgen_model)
        result = pandas.concat(result, axis=0, ignore_index=True, copy=False)
        result.drop_duplicates(inplace=True)
        result.reset_index(inplace=True, drop=True)
        return result


def main():
    """Function to be called when file executed via terminal."""
    print(__doc__)


if __name__ == "__main__":
    main()
