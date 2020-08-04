"""Commandline tool for creating a custom IGoR V(D)J model and valuating V(D)J sequences using an IGoR model."""


import logging
import os
import sys

import numpy

from shutil import copy2

from immuno_probs.cdr3.olga_container import OlgaContainer
from immuno_probs.model.default_models import get_default_model_file_paths
from immuno_probs.model.igor_interface import IgorInterface
from immuno_probs.util.cli import dynamic_cli_options
from immuno_probs.util.conversion import nucleotides_to_aminoacids
from immuno_probs.util.constant import get_config_data
from immuno_probs.util.io import read_separated_to_dataframe, read_fasta_as_dataframe, write_dataframe_to_separated, preprocess_separated_file, preprocess_reference_file, is_fasta, is_separated, copy_to_dir



class BuildAndEvaluate(object):
    """Commandline tool for creating custom IGoR V(D)J models and valuating V(D)J sequences using an IGoR model..

    Parameters
    ----------
    subparsers : argparse.ArgumentParser
        A subparser object for appending the tool's parser and options.

    Methods
    -------
    run(args)
        Uses the given Namespace commandline arguments to execute IGoR for creating a custom model and evaluating a sequence.

    """
    def __init__(self, subparsers):
        super(BuildAndEvaluate, self).__init__()
        self.logger = logging.getLogger(__name__)
        self.subparsers = subparsers
        self._add_options()

    
    def _add_options(self):
        """Function for adding the parser and options to the given ArgumentParser.

        Notes
        -----
            Uses the class constructor's subparser object for appending the tool's parser and options.

        """
        # Create the description and options for the parser.
        description = "Create a VDJ or VJ model by executing IGoR's commandline tool via a python subprocess using default " \
            "model parameters." \
            "Evaluate VDJ or VJ sequences given a custom IGoR model (or build-in) through IGoR's commandline " \
            "tool via python subprocess. Or evaluate CDR3 sequences with the model by using OLGA."
        parser_options = {
            '-seqs-build': {
                'metavar': '<fasta/separated>',
                'required': 'True',
                'type': 'str',
                'help': "An input FASTA or separated data file with sequences for training the model."
            },
            '-ref': {
                'metavar': ('<gene>', '<fasta>'),
                'type': 'str',
                'action': 'append',
                'nargs': 2,
                'required': 'True',
                'help': "A gene (V, D or J) followed by a reference genome FASTA file. Note: the FASTA reference genome files "
                        "needs to conform to IGMT annotation (separated by '|' character)."
            },
            '-type': {
                'type': 'str.lower',
                'choices': ['alpha', 'beta', 'light', 'heavy'],
                'required': 'True',
                'help': 'The type of model to create. (select one: %(choices)s).'
            },
            '-n-iter': {
                'type': 'int',
                'nargs': '?',
                'help': 'The number of inference iterations to perform when creating the model (default: {}).'
                        .format(get_config_data('BUILD', 'NUM_ITERATIONS', 'int'))
            },
            
            '-seqs-evaluate': {
                'metavar': '<fasta/separated>',
                'required': 'True',
                'type': 'str',
                'help': "An input FASTA or separated data file with sequences to evaluate."
            },
            '-anchor': {
                'metavar': ('<gene>', '<separated>'),
                'type': 'str',
                'action': 'append',
                'nargs': 2,
                'required': ('-cdr3' in sys.argv and '-custom-model' in sys.argv),
                'help': 'A gene (V or J) followed by a CDR3 anchor separated data file. Note: need to contain gene in the '
                        'first column, anchor index in the second and gene function in the third (required for -cdr3 and '
                        '-custom-model).'
            },
            '-cdr3': {
                'action': 'store_true',
                'help': 'If specified (True), CDR3 sequences should be evaluated, otherwise V(D)J sequences (default: {}).'
                        .format(get_config_data('EVALUATE', 'EVAL_CDR3', 'bool'))
            },
            '-use-allele': {
                'action': 'store_true',
                'help': "If specified (True), in combination with the '-cdr3' flag, the allele information from the gene "
                        "choice fields is used to calculate the generation probability (default: {})."
                        .format(get_config_data('EVALUATE', 'USE_ALLELE', 'bool'))
            }
        }

        # Add the options to the parser and return the updated parser.
        parser_tool = self.subparsers.add_parser('build_evaluate', help=description, description=description)
        parser_tool = dynamic_cli_options(parser=parser_tool, options=parser_options)




    @staticmethod
    def _copy_file_to_output(file, filename, directory):
        """Copies a txt file to the given directory.

        If the file already exists, a number will be appended to the filename.
        The given output directory is created recursively if it does not exist.

        Parameters
        ----------
        file : str
            A string path to the file to copy over to the directory
        filename : str
            Base filename for writting the file, excluding the extension.
        directory : str
            A directory path location to write the file to.

        Returns
        -------
        tuple
            Containing the output directory and the name of the file that has
            been written to disk.

        """
        # Check if the filename is unique, modify name if necessary.
        file_count = 1
        updated_filename = filename

        # Keep modifying the filename until it doesn't exist.
        while os.path.isfile(os.path.join(directory, updated_filename + '.txt')):
            updated_filename = str(filename) + '_' + str(file_count)
            file_count += 1

        # Copy input file to new location and return info.
        copy2(file, os.path.join(directory, updated_filename + '.txt'))
        return (directory, updated_filename + '.txt')



    def run(self, args, output_dir):
        """Function to execute the commandline tool.

        Parameters
        ----------
        args : Namespace
            Object containing our parsed commandline arguments.
        output_dir : str
            A directory path for writing output files to.

        """
        # Add general igor commands.
        self.logger.info('Setting up initial IGoR command (Build, 1/5)')
        command_list = []
        working_dir = get_config_data('COMMON', 'WORKING_DIR')
        command_list.append(['set_wd', working_dir])
        command_list.append(['threads', str(get_config_data('COMMON', 'NUM_THREADS', 'int'))])

        # Add sequence and file paths commands.
        self.logger.info('Processing genomic reference templates (Build, 2/5)')
        try:
            ref_list = ['set_genomic']
            for i in args.ref:
                filename = preprocess_reference_file(
                    os.path.join(working_dir, 'genomic_templates'),
                    copy_to_dir(working_dir, i[1], 'fasta'),
                    1
                )
                ref_list.append([i[0], filename])
            command_list.append(ref_list)
        except IOError as err:
            self.logger.error(str(err))
            return

        # Set the initial model parameters using a build-in model.
        self.logger.info('Setting initial model parameters (Build, 3/5)')
        if args.type in ['beta', 'heavy']:
            command_list.append([
                'set_custom_model',
                get_default_model_file_paths(name='human-t-beta')['parameters']
            ])
        elif args.type in ['alpha', 'light']:
            command_list.append([
                'set_custom_model',
                get_default_model_file_paths(name='human-t-alpha')['parameters']
            ])

        # Add the sequence command after pre-processing of the input file.
        self.logger.info('Pre-processing input sequence file (Build, 4/5)')
        try:
            if is_fasta(args.seqs_build):
                self.logger.info('FASTA input file extension detected')
                command_list.append([
                    'read_seqs',
                    copy_to_dir(working_dir, str(args.seqs_build), 'fasta')
                ])
            elif is_separated(args.seqs_build, get_config_data('COMMON', 'SEPARATOR')):
                self.logger.info('Separated input file type detected')
                try:
                    input_seqs = preprocess_separated_file(
                        os.path.join(working_dir, 'input'),
                        copy_to_dir(working_dir, str(args.seqs_build), 'csv'),
                        get_config_data('COMMON', 'SEPARATOR'),
                        ';',
                        get_config_data('COMMON', 'I_COL'),
                        [get_config_data('COMMON', 'NT_COL')]
                    )
                    command_list.append(['read_seqs', input_seqs])
                except (KeyError, ValueError) as err:
                    self.logger.error(
                        "Given input sequence file does not have a '%s' column",
                        get_config_data('COMMON', 'NT_COL'))
                    return
            else:
                self.logger.error(
                    'Given input sequence file could not be detected as '
                    'FASTA file or separated data type')
                return
        except (IOError, KeyError) as err:
            self.logger.error(str(err))
            return

        # Add alignment command and inference commands.
        self.logger.info('Adding additional variables to IGoR command (Build, 5/5)')
        command_list.append(['align', ['all']])
        if args.n_iter:
            command_list.append(['infer', ['N_iter', str(args.n_iter)]])
        else:
            command_list.append(['infer', [
                'N_iter', str(get_config_data('BUILD', 'NUM_ITERATIONS', 'int'))]])

        # Execute IGoR through command line and catch error code.
        self.logger.info('Executing IGoR (this might take a while)')
        try:
            igor_cline = IgorInterface(command=command_list)
            exit_code, _, stderr, _ = igor_cline.call()
            if exit_code != 0:
                self.logger.error(
                    "An error occurred during execution of IGoR command "
                    "(exit code %s):\n%s", exit_code, stderr)
                return
        except OSError as err:
            self.logger.error(str(err))
            return

        # Copy the output files to the output directory with prefix.
        try:
            self.logger.info('Writing model files to file system')
            output_prefix = get_config_data('COMMON', 'OUT_NAME')
            if not output_prefix:
                output_prefix = 'model'
            _, filename_1 = self._copy_file_to_output(
                file=os.path.join(working_dir, 'inference', 'final_marginals.txt'),
                filename='{}_marginals'.format(output_prefix),
                directory=output_dir)
            self.logger.info("Written '%s'", filename_1)
            _, filename_2 = self._copy_file_to_output(
                file=os.path.join(working_dir, 'inference', 'final_parms.txt'),
                filename='{}_params'.format(output_prefix),
                directory=output_dir)
            self.logger.info("Written '%s'", filename_2)
        except IOError as err:
            self.logger.error(str(err))
            return
            
            
            
            
            
        eval_cdr3 = get_config_data('EVALUATE', 'EVAL_CDR3', 'bool')
        if args.cdr3:
            eval_cdr3 = args.cdr3

        # If the given type of sequences evaluation is VDJ, use IGoR.
        if not eval_cdr3:

            # Add general IGoR commands.
            self.logger.info('Setting up initial IGoR command (Evaluate, 1/4)')
            command_list = []
            working_dir = get_config_data('COMMON', 'WORKING_DIR')
            command_list.append(['set_wd', working_dir])
            command_list.append(['threads', str(get_config_data('COMMON', 'NUM_THREADS', 'int'))])

            # Add the model (build-in or custom) command depending on given.
            self.logger.info('Processing genomic reference templates (Evaluate, 2/4)')
            try:
                model_type = args.type
                command_list.append([
                    'set_custom_model',
                    copy_to_dir(working_dir, filename_1, 'txt'),
                    copy_to_dir(working_dir, filename_2, 'txt'),
                ])
                ref_list = ['set_genomic']
                for i in args.ref:
                    filename = preprocess_reference_file(
                        os.path.join(working_dir, 'genomic_templates'),
                        copy_to_dir(working_dir, i[1], 'fasta'),
                        1
                    )
                    ref_list.append([i[0], filename])
                command_list.append(ref_list)
            except IOError as err:
                self.logger.error(str(err))
                return

            # Add the sequence command after pre-processing of the input file.
            self.logger.info('Pre-processing input sequence file (Evaluate, 3/4)')
            try:
                if is_fasta(args.seqs_evaluate):
                    self.logger.info('FASTA input file extension detected')
                    command_list.append([
                        'read_seqs',
                        copy_to_dir(working_dir, str(args.seqs_evaluate), 'fasta')
                    ])
                elif is_separated(args.seqs_evaluate, get_config_data('COMMON', 'SEPARATOR')):
                    self.logger.info('Separated input file type detected')
                    input_seqs = preprocess_separated_file(
                        os.path.join(working_dir, 'input'),
                        copy_to_dir(working_dir, str(args.seqs_evaluate), 'csv'),
                        get_config_data('COMMON', 'SEPARATOR'),
                        ';',
                        get_config_data('COMMON', 'I_COL'),
                        [get_config_data('COMMON', 'NT_COL')]
                    )
                    command_list.append(['read_seqs', input_seqs])
                else:
                    self.logger.error(
                        'Given input sequence file could not be detected as '
                        'FASTA file or separated data type')
                    return
            except (IOError, KeyError, ValueError) as err:
                self.logger.error(str(err))
                return

            # Add alignment and evealuation commands.
            self.logger.info('Adding additional variables to IGoR command (Evaluate, 4/4)')
            command_list.append(['align', ['all']])
            command_list.append(['evaluate'])
            command_list.append(['output', ['Pgen']])

            # Execute IGoR through command line and catch error code.
            self.logger.info('Executing IGoR (this might take a while)')
            try:
                igor_cline = IgorInterface(command=command_list)
                exit_code, _, stderr, _ = igor_cline.call()
                if exit_code != 0:
                    self.logger.error(
                        "An error occurred during execution of IGoR command "
                        "(exit code %s):\n%s", exit_code, stderr)
                    return
            except OSError as err:
                self.logger.error(str(err))
                return

            # Read in all data frame files, based on input file type.
            self.logger.info('Processing generation probabilities')
            try:
                if is_fasta(args.seqs_evaluate):
                    seqs_df = read_fasta_as_dataframe(
                        file=args.seqs_evaluate,
                        col=get_config_data('COMMON', 'NT_COL'))
                elif is_separated(args.seqs_evaluate, get_config_data('COMMON', 'SEPARATOR')):
                    seqs_df = read_separated_to_dataframe(
                        file=args.seqs_evaluate,
                        separator=get_config_data('COMMON', 'SEPARATOR'),
                        index_col=get_config_data('COMMON', 'I_COL'))
                full_pgen_df = read_separated_to_dataframe(
                    file=os.path.join(working_dir, 'output', 'Pgen_counts.csv'),
                    separator=';',
                    index_col='seq_index',
                    cols=['Pgen_estimate'])
                full_pgen_df.index.names = [get_config_data('COMMON', 'I_COL')]
                full_pgen_df.rename(
                    columns={'Pgen_estimate': get_config_data('COMMON', 'NT_P_COL')},
                    inplace=True)
                full_pgen_df.loc[:, get_config_data('COMMON', 'AA_P_COL')] = numpy.nan
            except (IOError, KeyError, ValueError) as err:
                self.logger.error(str(err))
                return

            # Insert amino acid sequence column if not existent.
            self.logger.info('Formatting output dataframe')
            if (get_config_data('COMMON', 'NT_COL') in seqs_df.columns
                    and not get_config_data('COMMON', 'AA_COL') in seqs_df.columns):
                seqs_df.insert(
                    seqs_df.columns.get_loc(get_config_data('COMMON', 'NT_COL')) + 1,
                    get_config_data('COMMON', 'AA_COL'), numpy.nan)
                seqs_df[get_config_data('COMMON', 'AA_COL')] = seqs_df[get_config_data('COMMON', 'NT_COL')].apply(nucleotides_to_aminoacids)

            # Merge IGoR generated sequence output dataframes.
            full_pgen_df = seqs_df.merge(full_pgen_df, left_index=True, right_index=True)

            # Write the pandas dataframe to a separated file.
            try:
                self.logger.info('Writing evaluated data to file system')
                output_filename = get_config_data('COMMON', 'OUT_NAME')
                if not output_filename:
                    output_filename = 'pgen_estimate_{}'.format(model_type)
                _, filename = write_dataframe_to_separated(
                    dataframe=full_pgen_df,
                    filename=output_filename,
                    directory=output_dir,
                    separator=get_config_data('COMMON', 'SEPARATOR'),
                    index_name=get_config_data('COMMON', 'I_COL'))
                self.logger.info("Written '%s'", filename)
            except IOError as err:
                self.logger.error(str(err))
                return

        # If the given type of sequences evaluation is CDR3, use OLGA.
        elif eval_cdr3:

            # Create the directory for the output files.
            working_dir = os.path.join(get_config_data('COMMON', 'WORKING_DIR'), 'output')
            if not os.path.isdir(working_dir):
                os.makedirs(os.path.join(get_config_data('COMMON', 'WORKING_DIR'), 'output'))

            # Load the model and create the sequence evaluator.
            self.logger.info('Loading the IGoR model files')
            try:
                model_type = args.type
                model = IgorLoader(model_type=model_type,
                                   model_params=args.custom_model[0],
                                   model_marginals=args.custom_model[1])
                separator = get_config_data('COMMON', 'SEPARATOR')
                for gene in args.anchor:
                    anchor_file = preprocess_separated_file(
                        os.path.join(working_dir, 'cdr3_anchors'),
                        str(gene[1]),
                        separator,
                        ','
                    )
                    model.set_anchor(gene=gene[0], file=anchor_file)
                model.initialize_model()
            except (TypeError, OSError, IOError, KeyError, ValueError) as err:
                self.logger.error(str(err))
                return

            # Based on input file type, load in input file.
            self.logger.info('Pre-processing input sequence file')
            try:
                if is_fasta(args.seqs_evaluate):
                    self.logger.info('FASTA input file extension detected')
                    seqs_df = read_fasta_as_dataframe(
                        file=args.seqs_evaluate,
                        col=get_config_data('COMMON', 'NT_COL'))
                elif is_separated(args.seqs_evaluate, get_config_data('COMMON', 'SEPARATOR')):
                    self.logger.info('Separated input file type detected')
                    seqs_df = read_separated_to_dataframe(
                        file=args.seqs_evaluate,
                        separator=get_config_data('COMMON', 'SEPARATOR'),
                        index_col=get_config_data('COMMON', 'I_COL'))
                else:
                    self.logger.error('Given input sequence file could not be detected as FASTA file or separated data type')
                    return
            except (IOError, KeyError, ValueError) as err:
                self.logger.error(str(err))
                return

            # Evaluate the sequences.
            self.logger.info('Evaluating sequences')
            try:
                use_allele = get_config_data('EVALUATE', 'USE_ALLELE', 'bool')
                if args.use_allele:
                    use_allele = args.use_allele
                seq_evaluator = OlgaContainer(
                    igor_model=model,
                    nt_col=get_config_data('COMMON', 'NT_COL'),
                    nt_p_col=get_config_data('COMMON', 'NT_P_COL'),
                    aa_col=get_config_data('COMMON', 'AA_COL'),
                    aa_p_col=get_config_data('COMMON', 'AA_P_COL'),
                    v_gene_choice_col=get_config_data('COMMON', 'V_GENE_CHOICE_COL'),
                    j_gene_choice_col=get_config_data('COMMON', 'J_GENE_CHOICE_COL'))
                cdr3_pgen_df = seq_evaluator.evaluate(
                    seqs=seqs_df,
                    num_threads=get_config_data('COMMON', 'NUM_THREADS', 'int'),
                    use_allele=use_allele,
                    default_allele=get_config_data('EVALUATE', 'DEFAULT_ALLELE'))

                # Merge IGoR generated sequence output dataframes.
                cdr3_pgen_df = seqs_df.merge(cdr3_pgen_df, left_index=True, right_index=True)
            except (TypeError, IOError) as err:
                self.logger.error(str(err))
                return

            # Write the pandas dataframe to a separated file.
            try:
                self.logger.info('Writing evaluated data to file system')
                output_filename = get_config_data('COMMON', 'OUT_NAME')
                if not output_filename:
                    output_filename = 'pgen_estimate_{}_CDR3'.format(model_type)
                _, filename = write_dataframe_to_separated(
                    dataframe=cdr3_pgen_df,
                    filename=output_filename,
                    directory=output_dir,
                    separator=get_config_data('COMMON', 'SEPARATOR'),
                    index_name=get_config_data('COMMON', 'I_COL'))
                self.logger.info("Written '%s'", filename)
            except IOError as err:
                self.logger.error(str(err))
                return
        
        
        
        
        
        
def main():
    """Function to be called when file executed via terminal."""
    print(__doc__)


if __name__ == "__main__":
    main()

