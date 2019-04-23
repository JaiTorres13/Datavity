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

#Load in the data from a CSV file that is comma seperated.
dataset = pd.read_csv('~/Desktop/Datavity_PL/Datavity/imports-85.csv',
                        sep=',', names=cols, header=None, encoding='latin-1')

print("done with import")

for fe in NUMERIC_FEATURES:
	dataset[fe] = pd.to_numeric(dataset[fe], errors='coerce')

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
TT_LPAREN   	= 'LPAREN'
TT_RPAREN   	= 'RPAREN'
TT_RBRACKET		= 'RBRACKET'
TT_LBRACKET		= 'LBRACKET'
TT_COMMA		= 'COMMA'
TT_EOF				= 'EOF'

KEYWORDS = [
	'VAR',
	'MEAN',
	'MIN',
	'MAX',
	'FEATURES',
	'CLEAN',
	'ZERO',
	'TRANSFORM',
	'RMSE'

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

#END OF PARSER
#############################################
#Intermediate Code
#############################################

def FEATURES():
	for feature in cols:
		print (feature)


def CLEAN(way):
	
	clean_data = dataset.copy()
	way_s = str(way).lower()
	if (way_s[-4:] == 'zero'):
		clean_data.fillna(0, inplace=True)
		
	elif (way_s[-4:] == 'mean'):
		for feature in NUMERIC_FEATURES:
			clean_data.fillna({feature: dataset[feature].mean()}, inplace=True)
	
	elif (way_s[-3:] == 'min'):
		for feature in NUMERIC_FEATURES:
			clean_data.fillna({feature: dataset[feature].min()}, inplace=True)

	else:
		for feature in NUMERIC_FEATURES:
			clean_data.fillna({feature: dataset[feature].max()}, inplace=True)
	
	ways = way_s[-4:]
	if (ways[0] == ':'):
		ways = ways[-3:]
	print('Replaced Nans with ' + ways)
	print(clean_data[1:20])
	return clean_data


#############################################################

# Linearly rescales to the range [0, 1]
def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = 1.0 * (max_val - min_val)
  return series.apply(lambda x:((x - min_val) / scale))

# Perform log scaling
def log_scale(series):
  return series.apply(lambda x:math.log(x+1.0))

# Clip all features to given min and max
def clip(series, clip_to_min, clip_to_max):
  # You need to modify this to actually do the clipping versus just returning
  # the series unchanged
  
  return series.apply(lambda x: clip_to_min if x<clip_to_min else clip_to_max if x>clip_to_max else x)



def TRANSFORM(feature_name):
  
  dataframe = dataset
  clip_min = -np.inf
  clip_max = np.inf
  plt.figure(figsize=(20, 5))
  plt.subplot(1, 3, 1)
  plt.title(feature_name)
  histogram = dataframe[feature_name].hist(bins=50)

  plt.subplot(1, 3, 2)
  plt.title("linear_scaling")
  scaled_features = dataset.copy()
  scaled_features[feature_name] = linear_scale(
      clip(dataframe[feature_name], clip_min, clip_max))
  histogram = scaled_features[feature_name].hist(bins=50)
  
  plt.subplot(1, 3, 3)
  plt.title("log scaling")
  log_normalized_features = dataset.copy()
  log_normalized_features[feature_name] = log_scale(dataframe[feature_name])
  histogram = log_normalized_features[feature_name].hist(bins=50)
  plt.show(histogram) 

 
def make_scatter_plot(dataframe, input_feature, target,
                      slopes=[], biases=[], model_names=[]):
  """ Creates a scatter plot of input_feature vs target along with the models.
  
  Args:
    dataframe: the dataframe to visualize
    input_feature: the input feature to be used for the x-axis
    target: the target to be used for the y-axis
    slopes: list of model weight (slope) 
    bias: list of model bias (same size as slopes)
    model_names: list of model_names to use for legend (same size as slopes)
  """      
  # Define some colors to use that go from blue towards red
  cmap = cm.get_cmap("spring")
  colors = [cmap(x) for x in np.linspace(0, 1, len(slopes))]
  
  # Generate the Scatter plot
  x = dataframe[input_feature]
  y = dataframe[target]
  plt.ylabel(target)
  plt.xlabel(input_feature)
  plt.scatter(x, y, color='black', label="")

  # Add the lines corresponding to the provided models
  for i in range (0, len(slopes)):
    y_0 = slopes[i] * x.min() + biases[i]
    y_1 = slopes[i] * x.max() + biases[i]
    plt.plot([x.min(), x.max()], [y_0, y_1],
             label=model_names[i], color=colors[i])
  if (len(model_names) > 0):
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  plt.show()



def RMSE(way, x, y):

	clean_data = CLEAN(way)
	way_s = str(way).lower()

	ways = way_s[-4:]
	if (ways[0] == ':'):
		ways = ways[-3:]

	# Plotting the new data set where rows where interpolated or estimated (filled)
	LABEL3 = "losses"
	INPUT_FEATURE3 = "price" 

	# model bias
	x3 = clean_data[INPUT_FEATURE3]
	y3 = clean_data[LABEL3]
	opt3 = np.polyfit(x3, y3, 1)
	y_pred3 = opt3[0] * x3 + opt3[1]
	opt_rmse3 = math.sqrt(metrics.mean_squared_error(y_pred3, y3))
	print("Root mean squared error for mean substitution") 
	print( opt_rmse3)
	slope3 = opt3[0]
	bias3 = opt3[1]

	LABEL3 = "losses"
	INPUT_FEATURE3 = "price" 
	plt.ylabel(LABEL3)
	plt.xlabel(INPUT_FEATURE3)
	plt.scatter(clean_data[INPUT_FEATURE3], clean_data[LABEL3], c='black')
	plt.title('Scatter Plot when Nan subtittuted by ' + ways)
	make_scatter_plot(clean_data,INPUT_FEATURE3, LABEL3,
					[slope3], [bias3], ["initial model"])
	plt.show()


#examples on how to run it in python
# FEATURES()
# TRANSFORM('losses')
# CLEAN('zero')
# RMSE('zero', 'prices', 'losses')

#examples on how to run it in datavity
# FEATURES()
# TRANSFORM(feature) where feature is any string
# CLEAN(ZERO) it can be also MIN, MAX, MEAN
# RMSE(ZERO, feature, feature) where feature is any string & first param can be also MIN, MAX, MEAN

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
