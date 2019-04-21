#######################################
# IMPORTS
#######################################

import strings_with_arrows

import string
import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
#import tensorflow as tf
#from tensorflow.contrib.learn.python.learn import learn_io, estimator


#######################################
# CONSTANTS
#######################################
from strings_with_arrows import string_with_arrows

DIGITS = '0123456789'
LETTERS = string.ascii_letters
LETTERS_DIGITS = LETTERS + DIGITS

#######################################
# DOWNLOAD DATASET & CREATE DATAFRAME
#######################################

# Set the output display to have one digit for decimal places, for display
# readability only and limit it to printing 15 rows.
pd.options.display.float_format = '{:.2f}'.format
pd.options.display.max_rows = 20

# Provide the names for the columns since the CSV file with the data does
# not have a header row.
cols = ['symboling', 'losses', 'make', 'fuel-type', 'aspiration', 'num-doors',
        'body-style', 'drive-wheels', 'engine-location', 'wheel-base',
        'length', 'width', 'height', 'weight', 'engine-type', 'num-cylinders',
        'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio',
        'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
NUMERIC_FEATURES = ['compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price',
			'symboling', 'losses', 'wheel-base', 'length', 'width', 'height', 'weight' , 'engine-size']
print("hello there")

#Load in the data from a CSV file that is comma seperated.
car_data = pd.read_csv('~/Desktop/Datavity_PL/Datavity/DataSamples/imports-85.csv',
                        sep=',', names=cols, header=None, encoding='latin-1')

print("done with import")

for fe in NUMERIC_FEATURES:
	car_data[fe] = pd.to_numeric(car_data[fe], errors='coerce')

#######################################
# ERRORS
#######################################

class Error:
	def __init__(self, pos_start, pos_end, error_name, details):
		self.pos_start = pos_start
		self.pos_end = pos_end
		self.error_name = error_name
		self.details = details
	
	def as_string(self):
		result = f'{self.error_name}: {self.details}\n'
		result += f'File {self.pos_start.fn}, line {self.pos_start.ln + 1}'
		result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
		return result

class IllegalCharError(Error):
	def __init__(self, pos_start, pos_end, details):
		super().__init__(pos_start, pos_end, 'Illegal Character', details)

class ExpectedCharError(Error):
	def __init__(self, pos_start, pos_end, details):
		super().__init__(pos_start, pos_end, 'Expected Character', details)

class InvalidSyntaxError(Error):
	def __init__(self, pos_start, pos_end, details=''):
		super().__init__(pos_start, pos_end, 'Invalid Syntax', details)

class RTError(Error):
	def __init__(self, pos_start, pos_end, details, context):
		super().__init__(pos_start, pos_end, 'Runtime Error', details)
		self.context = context

	def as_string(self):
		result  = self.generate_traceback()
		result += f'{self.error_name}: {self.details}'
		result += '\n\n' + string_with_arrows(self.pos_start.ftxt, self.pos_start, self.pos_end)
		return result

	def generate_traceback(self):
		result = ''
		pos = self.pos_start
		ctx = self.context

		while ctx:
			result = f'  File {pos.fn}, line {str(pos.ln + 1)}, in {ctx.display_name}\n' + result
			pos = ctx.parent_entry_pos
			ctx = ctx.parent

		return 'Traceback (most recent call last):\n' + result

#######################################
# POSITION
#######################################

class Position:
	def __init__(self, idx, ln, col, fn, ftxt):
		self.idx = idx
		self.ln = ln
		self.col = col
		self.fn = fn
		self.ftxt = ftxt

	def advance(self, current_char=None):
		self.idx += 1
		self.col += 1

		if current_char == '\n':
			self.ln += 1
			self.col = 0

		return self

	def copy(self):
		return Position(self.idx, self.ln, self.col, self.fn, self.ftxt)

#######################################
# TOKENS
#######################################

TT_INT				= 'INT'
TT_FLOAT    	= 'FLOAT'
TT_IDENTIFIER	= 'IDENTIFIER'
TT_KEYWORD		= 'KEYWORD'
TT_PLUS     	= 'PLUS'
TT_MINUS    	= 'MINUS'
TT_MUL      	= 'MUL'
TT_DIV      	= 'DIV'
TT_POW				= 'POW'
TT_EQ					= 'EQ'
TT_LPAREN   	= 'LPAREN'
TT_RPAREN   	= 'RPAREN'
TT_RBRACKET		= 'RBRACKET'
TT_LBRACKET		= 'LBRACKET'
TT_COMMA		= 'COMMA'
TT_EOF				= 'EOF'

KEYWORDS = [
	'VAR',
	'AND',
	'OR',
	'NOT',
	'IF',
	'THEN',
	'ELIF',
	'ELSE',
	'MEAN',
	'MIN',
	'MAX',
	'LOG',
	'LINEAR',
	'FEATURES',
	'CLEAN',
	'DESCRIBE',
	'CLEAN',
	'ZERO',
	'CLIPPING'

]

class Token:
	def __init__(self, type_, value=None, pos_start=None, pos_end=None):
		self.type = type_
		self.value = value

		if pos_start:
			self.pos_start = pos_start.copy()
			self.pos_end = pos_start.copy()
			self.pos_end.advance()

		if pos_end:
			self.pos_end = pos_end.copy()

	def matches(self, type_, value):
		return self.type == type_ and self.value == value
	
	def __repr__(self):
		if self.value: return f'{self.type}:{self.value}'
		return f'{self.type}'

#######################################
# LEXER
#######################################

class Lexer:
	def __init__(self, fn, text):
		self.fn = fn
		self.text = text
		self.pos = Position(-1, 0, -1, fn, text)
		self.current_char = None
		self.advance()
	
	def advance(self):
		self.pos.advance(self.current_char)
		self.current_char = self.text[self.pos.idx] if self.pos.idx < len(self.text) else None

	def make_tokens(self):
		tokens = []

		while self.current_char != None:
			if self.current_char in ' \t':
				self.advance()
			elif self.current_char in DIGITS:
				tokens.append(self.make_number())
			elif self.current_char in LETTERS:
				tokens.append(self.make_identifier())
			elif self.current_char == '+':
				tokens.append(Token(TT_PLUS, pos_start=self.pos))
				self.advance()
			elif self.current_char == '-':
				tokens.append(Token(TT_MINUS, pos_start=self.pos))
				self.advance()
			elif self.current_char == '*':
				tokens.append(Token(TT_MUL, pos_start=self.pos))
				self.advance()
			elif self.current_char == '/':
				tokens.append(Token(TT_DIV, pos_start=self.pos))
				self.advance()
			elif self.current_char == '^':
				tokens.append(Token(TT_POW, pos_start=self.pos))
				self.advance()
			elif self.current_char == '(':
				tokens.append(Token(TT_LPAREN, pos_start=self.pos))
				self.advance()
			elif self.current_char == ')':
				tokens.append(Token(TT_RPAREN, pos_start=self.pos))
				self.advance()
			elif self.current_char == '[':
				tokens.append(Token(TT_LBRACKET, pos_start=self.pos))
				self.advance()
			elif self.current_char == ']':
				tokens.append(Token(TT_RBRACKET, pos_start=self.pos))
				self.advance()
			elif self.current_char == ',':
				tokens.append(Token(TT_COMMA, pos_start=self.pos))
				self.advance()
			else:
				pos_start = self.pos.copy()
				char = self.current_char
				self.advance()
				return [], IllegalCharError(pos_start, self.pos, "'" + char + "'")

		tokens.append(Token(TT_EOF, pos_start=self.pos))
		return tokens, None


	def make_identifier(self):
		id_str = ''
		pos_start = self.pos.copy()

		while self.current_char != None and self.current_char in LETTERS_DIGITS + '_':
			id_str += self.current_char
			self.advance()

		tok_type = TT_KEYWORD if id_str in KEYWORDS else TT_IDENTIFIER
		return Token(tok_type, id_str, pos_start, self.pos)
	
	def make_number(self):
		num_str = ''
		dot_count = 0
		pos_start = self.pos.copy()

		while self.current_char != None and self.current_char in DIGITS + '.':
			if self.current_char == '.':
				if dot_count == 1: break
				dot_count += 1
			num_str += self.current_char
			self.advance()

		if dot_count == 0:
			return Token(TT_INT, int(num_str), pos_start, self.pos)
		else:
			return Token(TT_FLOAT, float(num_str), pos_start, self.pos)


#######################################
# NODES
#######################################

class FeatureNode:
	def __init__(self, tok):
		self.tok = tok

		self.pos_start = self.tok.pos_start
		self.pos_end = self.tok.pos_end


#######################################
# PARSE RESULT
#######################################

class ParseResult:
	def __init__(self):
		self.error = None
		self.node = None
		self.advance_count = 0

	def register_advancement(self):
		self.advance_count += 1

	def register(self, res):
		self.advance_count += res.advance_count
		if res.error: self.error = res.error
		return res.node

	def success(self, node):
		self.node = node
		return self

	def failure(self, error):
		if not self.error or self.advance_count == 0:
			self.error = error
		return self

#######################################
# PARSER
#######################################

class Parser:
	def __init__(self, tokens):
		self.tokens = tokens
		self.tok_idx = -1
		self.advance()

	def advance(self, ):
		self.tok_idx += 1
		if self.tok_idx < len(self.tokens):
			self.current_tok = self.tokens[self.tok_idx]
		return self.current_tok

	def parse(self):
		res = self.datavity()
		if not res.error and self.current_tok.type != TT_EOF:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected KEYWORD"
			))
		return res

	###################################


	def datavity(self):
		res = ParseResult()
		if self.current_tok.matches(TT_KEYWORD, 'FEATURES'):
			res.register_advancement()
			self.advance()

			if self.current_tok.type != TT_LPAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected '('"
				))

			res.register_advancement()
			self.advance()

			if self.current_tok.type != TT_RPAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected ')'"
				))
			
			FEATURES()
			res.register_advancement()
			self.advance()

		if self.current_tok.matches(TT_KEYWORD, 'CLEAN'):
			res.register_advancement()
			self.advance()

			if self.current_tok.type != TT_LPAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected '('"
				))

			res.register_advancement()
			self.advance()


			if not (self.current_tok.matches(TT_KEYWORD, 'MIN') or self.current_tok.matches(TT_KEYWORD, 'MEAN')
				or self.current_tok.matches(TT_KEYWORD, 'MAX') or self.current_tok.matches(TT_KEYWORD, 'ZERO')):
					return res.failure(InvalidSyntaxError(
						self.current_tok.pos_start, self.current_tok.pos_end,
						"Expected 'MIN', 'MAX', 'MEAN', 'ZERO'"
					))

			method = self.current_tok
			res.register_advancement()
			self.advance()

			if self.current_tok.type != TT_RPAREN:
				return res.failure(InvalidSyntaxError(
					self.current_tok.pos_start, self.current_tok.pos_end,
					"Expected ')'"
				))
			
			
			CLEAN(method)
			res.register_advancement()
			self.advance()

		if res.error:
			return res.failure(InvalidSyntaxError(
				self.current_tok.pos_start, self.current_tok.pos_end,
				"Expected 'FEATURE', identifier"
			))
		return res.success(FeatureNode(self.current_tok))


def FEATURES():
	for feature in cols:
				print (feature)

def CLEAN(way):
	clean_data = car_data.copy()
	way_s = str(way).lower()
	print(way_s)
	if (way_s[-4:] == 'zero'):
		clean_data.fillna(0, inplace=True)
		
	elif (way_s[-4:] == 'mean'):
		for feature in NUMERIC_FEATURES:
			clean_data.fillna({feature: car_data[feature].mean()}, inplace=True)
	
	elif (way_s[-3:] == 'min'):
		for feature in NUMERIC_FEATURES:
			clean_data.fillna({feature: car_data[feature].min()}, inplace=True)

	else:
		for feature in NUMERIC_FEATURES:
			clean_data.fillna({feature: car_data[feature].max()}, inplace=True)
			# car_data.fillna(car_data[feature].mean(), inplace=True)
	
	print(clean_data[1:20])

#END OF PARSER

#######################################
# RUNTIME RESULT
#######################################

# class RTResult:
# 	def __init__(self):
# 		self.value = None
# 		self.error = None

# 	def register(self, res):
# 		if res.error: self.error = res.error
# 		return res.value

# 	def success(self, value):
# 		self.value = value
# 		return self

# 	def failure(self, error):
# 		self.error = error
# 		return self

#######################################
# VALUES
#######################################

# class Number:
# 	def __init__(self, value):
# 		self.value = value
# 		self.set_pos()
# 		self.set_context()

# 	def set_pos(self, pos_start=None, pos_end=None):
# 		self.pos_start = pos_start
# 		self.pos_end = pos_end
# 		return self

# 	def set_context(self, context=None):
# 		self.context = context
# 		return self

# 	def copy(self):
# 		copy = Number(self.value)
# 		copy.set_pos(self.pos_start, self.pos_end)
# 		copy.set_context(self.context)
# 		return copy

# 	def is_true(self):
# 		return self.value != 0
	
# 	def __repr__(self):
# 		return str(self.value)

#######################################
# CONTEXT
#######################################

# class Context:
# 	def __init__(self, display_name, parent=None, parent_entry_pos=None):
# 		self.display_name = display_name
# 		self.parent = parent
# 		self.parent_entry_pos = parent_entry_pos
# 		self.symbol_table = None




#######################################
# RUN
#######################################


def run(fn, text):
	# Generate tokens
	lexer = Lexer(fn, text)
	tokens, error = lexer.make_tokens()
	if error: return None, error
	
	# Generate AST
	parser = Parser(tokens)
	ast = parser.parse()
	if ast.error: return None, ast.error
