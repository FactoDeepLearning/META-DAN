from dar.dataset_format.bressay import add_bressay_formatter_parser
from dar.dataset_format.casia2 import add_casia2_formatter_parser
from dar.dataset_format.eparchos import add_eparchos_formatter_parser
from dar.dataset_format.esposalles import add_esposalles_formatter_parser
from dar.dataset_format.iam import add_iam_formatter_parser
from dar.dataset_format.maurdor import add_maurdor_formatter_parser
from dar.dataset_format.read2016 import add_read2016_formatter_parser
from dar.dataset_format.rimes2009 import add_rimes2009_formatter_parser
from dar.dataset_format.rimes2024 import add_rimes2024_formatter_parser
from dar.dataset_format.scribblelens import add_scribblelens_formatter_parser


def add_format_dataset_parser(subcommands):
    parser = subcommands.add_parser("format")
    subparsers = parser.add_subparsers(metavar="subcommand", dest="dataset_name")

    add_bressay_formatter_parser(subparsers)
    add_casia2_formatter_parser(subparsers)
    add_eparchos_formatter_parser(subparsers)
    add_esposalles_formatter_parser(subparsers)
    add_iam_formatter_parser(subparsers)
    add_maurdor_formatter_parser(subparsers)
    add_read2016_formatter_parser(subparsers)
    add_rimes2009_formatter_parser(subparsers)
    add_rimes2024_formatter_parser(subparsers)
    add_scribblelens_formatter_parser(subparsers)

