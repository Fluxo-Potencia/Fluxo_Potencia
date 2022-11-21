# na janela de comandos inserir:
# set FLASK_APP=Fluxo_Potencia
# set FLASK_ENV=development
# flask run

# abrir no navegador o site: 127.0.0.1:5000

# importing two methods from flask, "Flask" and "render_template"
from flask import Flask, render_template, request
import math
import numpy as np


# Creating the web application

Flask_App = Flask(__name__)

V1 = 1.0
t1 = 0
r = 0.05
x = 0.20
bsh = 0.1
P2 = 0.8
Q2 = 0.2

V1_3 = 1.02
t1_3 = 0
r12 = 0.0385
x12 = 0.1923
r13 = 0.0294
x13 = 0.1176
r23 = 0.0385
x23 = 0.1923
bsh12= 0
bsh13= 0
bsh23= 0
P2_3 = 1.5
Q2_3 = 0.6
V3 = 1.01
P3 = 0.25

@Flask_App.route('/', methods=['GET'])
def index():
	""" Display the index page accesible at '/' """
	return render_template(
		'index.html',
		input_V1 = V1,
		input_t1 = t1,
		input_r = r,
		input_x = x,
		input_bsh = bsh,
		input_P2 = P2,
		input_Q2 = Q2,
		result = "",
		calculation_success = True
	)

@Flask_App.route('/operation_result/', methods=['POST'])
def operation_result():
	"""	Route where we send calculator form input"""
	error = None
	result = None
	V1z = request.form['Input_V1']
	t1z = request.form['Input_t1']
	t1z = request.form['Input_t1']
	rz = request.form['Input_r']
	xz = request.form['Input_x']
	bshz = request.form['Input_bsh']
	P2z = request.form['Input_P2']
	Q2z = request.form['Input_Q2']

	try:
		V1 = float(V1z.replace(',','.'))
		t1 = float(t1z.replace(',','.'))*math.pi/180
		r = float(rz.replace(',','.'))
		x = float(xz.replace(',','.'))
		bsh = float(bshz.replace(',','.'))
		P2 = float(P2z.replace(',','.'))
		Q2 = float(Q2z.replace(',','.'))


		z2 = r**2 + x**2
		g = r/z2
		b = -x/z2
		A = g**2 + (b+bsh/2)**2
		B = 2*P2*g - 2*Q2*(b+bsh/2) - V1**2*(g**2+b**2)
		C = P2**2 + Q2**2

		D = B**2 - 4*A*C
		if D >= 0:
			V2 = math.sqrt( (-B+math.sqrt(D)) / (2*A) )
			t2 = math.asin( (b*P2 + g*Q2 -g*bsh/2*V2**2) / (V1*V2*(g**2+b**2)) )
			P1 = g*V1**2 - V1*V2*( g*math.cos(-t2)+ b*math.sin(-t2))
			Q1 = -(b+bsh/2)*V1**2 - V1*V2*( g*math.sin(-t2) - b*math.cos(-t2) )
			t2 = (t2 + t1)*180/math.pi
			# self.node1.vD[2].setPlainText(f'{P1g:.4f}pu')
			# self.node1.vD[3].setPlainText(f'{Q1g:.4f}pu')
			# self.node2.vD[2].setPlainText(f'{V2:.4f}pu')
			return render_template(
				'index.html',
				input_V1 = V1z,
				input_t1 = t1z,
				input_r = rz,
				input_x = xz,
				input_bsh = bshz,
				input_P2 = P2z,
				input_Q2 = Q2z,
				result_V2 = str(int(1e4*V2)/1e4).replace('.',',') + ' p.u.',
				result_t2 = str(int(1e2*t2)/1e2).replace('.',',') + u'\N{DEGREE SIGN}',
				result_P1 = str(int(1e4*P1)/1e4).replace('.',',') + ' p.u.',
				result_Q1 = str(int(1e4*Q1)/1e4).replace('.',',') + ' p.u.',
				calculation_success = True
			)

		else:
			V2 = 'NaN'
			t2 = 'NaN'
			V2 = 'NaN'
			t2 = 'NaN'
			P1 = 'NaN'
			Q1 = 'NaN'
			return render_template(
				'index.html',
				input_V1 = V1z,
				input_t1 = t1z,
				input_r = rz,
				input_x = xz,
				input_bsh = bshz,
				input_P2 = P2z,
				input_Q2 = Q2z,
				result_V2 = 'NaN',
				result_t2 = 'NaN',
				result_P1 = 'NaN',
				result_Q1 = 'NaN',
				calculation_success = True
			)
			# self.node1.vD[2].setPlainText('NaN')
			# self.node1.vD[3].setPlainText('NaN')
			# self.node2.vD[2].setPlainText('NaN')
			# self.node2.vD[3].setPlainText('NaN')



	except ZeroDivisionError:
		return render_template(
			'index.html',
			input_V1 = V1z,
			input_t1 = t1z,
			input_r = rz,
			input_x = xz,
			input_bsh = bshz,
			input_P2 = P2z,
			input_Q2 = Q2z,
			result = "Bad Input",
			calculation_success = False,
			error="You cannot divide by zero"
		)

	except ValueError:
		return render_template(
			'index.html',
			input_V1 = V1z,
			input_t1 = t1z,
			input_r = rz,
			input_x = xz,
			input_bsh = bshz,
			input_P2 = P2z,
			input_Q2 = Q2z,
			result_V2 = V2z,
			result_t2 = t2z,
			result = "Bad Input",
			calculation_success = False,
			error="Cannot perform numeric operations with provided input"
		)


@Flask_App.route('/inicial/', methods=['POST'])
def inicial():
	"""	Route where we send caclulator form input"""

	return render_template(
		'index.html',
		input_V1 = V1,
		input_t1 = t1,
		input_r = r,
		input_x = x,
		input_bsh = bsh,
		input_P2 = P2,
		input_Q2 = Q2,
		result = "",
		calculation_success = True
	)


@Flask_App.route('/sist3/', methods=['POST'])
def sist3():
	"""	Route where we send caclulator form input"""


	return render_template(
		'index3.html',
		input_V1 = V1_3,
		input_t1 = t1_3,
		input_r12 = r12,
		input_x12 = x12,
		input_r13 = r13,
		input_x13 = x13,
		input_r23 = r23,
		input_x23 = x23,
		input_bsh12 = bsh12,
		input_bsh13 = bsh13,
		input_bsh23 = bsh23,
		input_P2 = P2_3,
		input_Q2 = Q2_3,
		input_V3 = V3,
		input_P3 = P3,
		result = "",
		calculation_success = True
	)

@Flask_App.route('/operation_result_3/', methods=['POST'])
def operation_result_3():
	"""	Route where we send calculator form input"""
	error = None
	result = None
	V1z = request.form['Input_V1']
	t1z = request.form['Input_t1']
	t1z = request.form['Input_t1']
	r12z = request.form['Input_r12']
	x12z = request.form['Input_x12']
	r13z = request.form['Input_r13']
	x13z = request.form['Input_x13']
	r23z = request.form['Input_r23']
	x23z = request.form['Input_x23']
	bsh12z = request.form['Input_bsh12']
	bsh13z = request.form['Input_bsh13']
	bsh23z = request.form['Input_bsh23']
	P2z = request.form['Input_P2']
	Q2z = request.form['Input_Q2']
	V3z = request.form['Input_V3']
	P3z = request.form['Input_P3']

	try:
		V1 = float(V1z.replace(',','.'))
		t1 = float(t1z.replace(',','.'))*math.pi/180
		r12 = float(r12z.replace(',','.'))
		x12 = float(x12z.replace(',','.'))
		r13 = float(r13z.replace(',','.'))
		x13 = float(x13z.replace(',','.'))
		r23 = float(r23z.replace(',','.'))
		x23 = float(x23z.replace(',','.'))
		bsh12 = float(bsh12z.replace(',','.'))*1j
		bsh13 = float(bsh13z.replace(',','.'))*1j
		bsh23 = float(bsh23z.replace(',','.'))*1j
		P2 = float(P2z.replace(',','.'))
		Q2 = float(Q2z.replace(',','.'))
		V3 = float(V3z.replace(',','.'))
		P3 = float(P3z.replace(',','.'))


		y12 = 1/(r12 + x12*1j)
		y13 = 1/(r13 + x13*1j)
		y23 = 1/(r23 + x23*1j)
		Y = np.matrix([[y12+y13+bsh12/2+bsh13/2, -y12, -y13],[-y12, y12+y23+bsh12/2+bsh23/2, -y23],[-y13, -y23, y13+y23+bsh13/2+bsh23/2]])
		G = np.real(Y)
		B = np.imag(Y)
		V2 = 1.0
		t2 = 0
		t3 = 0
		for h in range(0, 10):
			Q3 = V3 * ( V1*(G[2,0]*np.sin(t3-t1)-B[2,0]*np.cos(t3-t1)) + V2*(G[2,1]*np.sin(t3-t2)-B[2,1]*np.cos(t3-t2)) + V3*(G[2,2]*np.sin(t3-t3)-B[2,2]*np.cos(t3-t3)) )
			V2cal = 1/Y[1,1] * ( np.conj(-P2-Q2*1j)/np.conj(V2*np.exp(1j*t2)) - Y[1,0]*V1*np.exp(1j*t1) - Y[1,2]*V3*np.exp(1j*t3) )
			V3cal = 1/Y[2,2] * ( np.conj(P3+Q3*1j)/np.conj(V3*np.exp(1j*t3)) - Y[2,0]*V1*np.exp(1j*t1) - Y[2,1]*V2*np.exp(1j*t2) )
			V2 = abs(V2cal)
			t2 = np.arctan(np.imag(V2cal)/np.real(V2cal))
			t3 = np.arctan(np.imag(V3cal)/np.real(V3cal))

		P1 = V1 * ( V1*(G[0,0]*np.cos(t1-t1)+B[0,0]*np.sin(t1-t1)) + V2*(G[0,1]*np.cos(t1-t2)+B[0,1]*np.sin(t1-t2)) + V3*(G[0,2]*np.cos(t1-t3)+B[0,2]*np.sin(t1-t3)) )
		Q1 = V1 * ( V1*(G[0,0]*np.sin(t1-t1)-B[0,0]*np.cos(t1-t1)) + V2*(G[0,1]*np.sin(t1-t2)-B[0,1]*np.cos(t1-t2)) + V3*(G[0,2]*np.sin(t1-t3)-B[0,2]*np.cos(t1-t3)) )

		t2 = (t2 + t1)*180/math.pi
		t3 = (t3 + t1)*180/math.pi

		return render_template(
			'index3.html',
			input_V1 = V1z,
			input_t1 = t1z,
			input_r12 = r12z,
			input_x12 = x12z,
			input_r13 = r13z,
			input_x13 = x13z,
			input_r23 = r23z,
			input_x23 = x23z,
			input_bsh12 = bsh12z,
			input_bsh13 = bsh13z,
			input_bsh23 = bsh23z,
			input_P2 = P2z,
			input_Q2 = Q2z,
			input_V3 = V3z,
			input_P3 = P3z,
			result_V2 = str(int(1e4*V2)/1e4).replace('.',',') + ' p.u.',
			result_t2 = str(int(1e2*t2)/1e2).replace('.',',') + u'\N{DEGREE SIGN}',
			result_t3 = str(int(1e2*t3)/1e2).replace('.',',') + u'\N{DEGREE SIGN}',
			result_Q3 = str(int(1e4*Q3)/1e4).replace('.',',') + ' p.u.',
			result_P1 = str(int(1e4*P1)/1e4).replace('.',',') + ' p.u.',
			result_Q1 = str(int(1e4*Q1)/1e4).replace('.',',') + ' p.u.',
			calculation_success = True
		)

		# else:
			# V2 = 'NaN'
			# t2 = 'NaN'
			# V2 = 'NaN'
			# t2 = 'NaN'
			# t3 = 'NaN'
			# P3 = 'NaN'
			# P1 = 'NaN'
			# Q1 = 'NaN'
			# return render_template(
				# 'index3.html',
				# input_V1 = V1z,
				# input_t1 = t1z,
				# input_r12 = r12z,
				# input_x12 = x12z,
				# input_r13 = r13z,
				# input_x13 = x13z,
				# input_r23 = r23z,
				# input_x23 = x23z,
				# input_bsh12 = bsh12z,
				# input_bsh13 = bsh13z,
				# input_bsh23 = bsh23z,
				# input_P2 = P2z,
				# input_Q2 = Q2z,
				# input_V3 = V3z,
				# input_P3 = P3z,
				# result_V2 = 'NaN',
				# result_t2 = 'NaN',
				# result_t3 = 'NaN',
				# result_Q3 = 'NaN',
				# result_P1 = 'NaN',
				# result_Q1 = 'NaN',
				# calculation_success = True
			# )
			# self.node1.vD[2].setPlainText('NaN')
			# self.node1.vD[3].setPlainText('NaN')
			# self.node2.vD[2].setPlainText('NaN')
			# self.node2.vD[3].setPlainText('NaN')



	except ZeroDivisionError:
		return render_template(
			'index3.html',
			input_V1 = V1z,
			input_t1 = t1z,
			input_r12 = r12z,
			input_x12 = x12z,
			input_r13 = r13z,
			input_x13 = x13z,
			input_r23 = r23z,
			input_x23 = x23z,
			input_bsh12 = bsh12z,
			input_bsh13 = bsh13z,
			input_bsh23 = bsh23z,
			input_P2 = P2z,
			input_Q2 = Q2z,
			input_V3 = V3z,
			input_P3 = P3z,
			result = "Bad Input",
			calculation_success = False,
			error="You cannot divide by zero"
		)

	except ValueError:
		return render_template(
			'index3.html',
			input_V1 = V1z,
			input_t1 = t1z,
			input_r = rz,
			input_x = xz,
			input_bsh12 = bsh12z,
			input_bsh13 = bsh13z,
			input_bsh23 = bsh23z,
			input_P2 = P2z,
			input_Q2 = Q2z,
			result_V2 = V2z,
			result_t2 = t2z,
			result = "Bad Input",
			calculation_success = False,
			error="Cannot perform numeric operations with provided input"
		)


@Flask_App.route('/sist3/', methods=['POST'])
def inicial3():
	"""	Route where we send caclulator form input"""

	return render_template(
		'index3.html',
		input_V1 = V1z,
		input_t1 = t1z,
		input_r12 = r12z,
		input_x12 = x12z,
		input_r13 = r13z,
		input_x13 = x13z,
		input_r23 = r23z,
		input_x23 = x23z,
		input_bsh12 = bsh12z,
		input_bsh13 = bsh13z,
		input_bsh23 = bsh23z,
		input_P2 = P2z,
		input_Q2 = Q2z,
		input_V3 = V3z,
		input_P3 = P3z,
		result = "",
		calculation_success = True
	)
