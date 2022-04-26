from ddp_args_helper import parser_args

def do_something_outside(rank):
	print("Re-imported parser_args, time to see if something funky happened. --> Local Rank: {} | parser_args.gpu={}, parser_args.name={}".format(rank, parser_args.gpu, parser_args.name))
	return -1
