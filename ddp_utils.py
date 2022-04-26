from args_helper import parser_args

def do_something(rank):
	print("Re-imported parser_args, time to see if something funky happened.")
	print("Local Rank: {} | parser_args.gpu={}, parser_args.name={}".format(parser_args.gpu, parser_args.name))
	return -1