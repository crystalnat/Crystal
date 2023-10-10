import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import pandas as pd
from bokeh.plotting import figure
from scipy.optimize import curve_fit

st.title('Selamat Datang, webApp ini masih dalam pengembangan')

with st.sidebar :
    A = option_menu ('Fungsi', ['Fungsi Linear',
                                        'Fungsi NonLinear',
                                        'Fungsi Eksponensial'],
                                        default_index=0)

if (A == 'Fungsi Linear'):
    data = st.file_uploader("Masukkan file excel :", type=["xlsx"])
    if data is not None:
        dp = pd.read_excel(data, engine='openpyxl')
        st.write(dp)

        
        data1 = dp[dp.columns[0]].values
        data2 = dp[dp.columns[1]]

        o1 = []
        o2 = []
        for i in range (len(data1)-1) :
            ol1 = data1 [i+1] - data1 [i]
            ol2 = data2 [i+1] - data2 [i]
            o1.append(ol1)
            o2.append(ol2)

        m = []
        for i in range(len(o2)) :
            m1 = o2[i] / o1[i]
            m.append(m1)
        mr = sum(m) / len(m)

        data1r = sum(data1) / len(data1)
        data2r = sum(data2) / len(data2)

        C = data2r - mr*data1r

        r2 = []
        for i in range(len(data1)):
            r = mr*data1[i] + C
            r2.append(r)

        error = []
        for i in range(len(r2)):
            e = data2[i] - r2[i]
            error.append(e)

        error2 = []
        for i in range(len(error)):
            e2 = error[i]**2
            error2.append(e2)
            SE = sum(error2)
            MSE = sum(error2)/len(error2)
            RMSE = np.sqrt(MSE)

        st.write("Persamaan fungsinya adalah :")
        st.write("y =",mr,"x +", C)
        st.write("SE =",SE)
        st.write("MSE =", MSE)
        st.write("RMSE =", RMSE)

 
        x = data1
        y = data2

        x1 = [min(data1), max(data1)]
        y1 = [mr * i + C for i in x1]

        p = figure(
            title='Regresi Linear',
            x_axis_label=dp.columns[0],
            y_axis_label=dp.columns[1])
    
        p.scatter(x, y, marker='circle', size=6, fill_color='blue')
        p.line(x, y, legend_label='Data', line_width=2)
        p.line(x1, y1, legend_label='Regresi Linear', line_width=2, line_color='red')

        st.bokeh_chart(p, use_container_width=True)

if (A == 'Fungsi NonLinear') :
        datanl = st.file_uploader("Masukkan file excel :", type=["xlsx"])
        if datanl is not None:
            dp1 = pd.read_excel(datanl, engine='openpyxl')
            st.write(dp1)

            x2 = np.array(dp1[dp1.columns[0]], float)
            y2 = np.array(dp1[dp1.columns[1]], float)


            a = st.slider('Masukkan orde', 0, 50)
            st.write("Orde", a)

            X = np.vander(x2, a+1, increasing=True)
            koefisien = np.linalg.inv(X.T @ X) @ X.T @ y2
                
            persamaan = "fungsi NonLinear P(x) = "
            for i, j in enumerate(koefisien[::-1]):
                persamaan += f"({j:.2f}) * x^{i} + "
            persamaan = persamaan[:-3] 
            st.write(persamaan)

            x_fit = np.linspace(x2.min(), 80)
            y_fit = np.polyval(koefisien[::-1], x_fit)

            p1 = figure(
                title='Regresi NonLinear',
                x_axis_label=dp1.columns[0],
                y_axis_label=dp1.columns[1])
    
            p1.scatter(x2, y2, marker='circle', size=6, fill_color='blue')
            p1.line(x2, y2, legend_label='Data', line_width=2)
            p1.line(x_fit, y_fit, legend_label='Regresi NonLinear', line_width=2, line_color='red')

            st.bokeh_chart(p1, use_container_width=True)

if (A == 'Fungsi Eksponensial') :
        datae = st.file_uploader("Masukkan file excel :", type=["xlsx"])
        if datae is not None:
            dp2 = pd.read_excel(datae, engine='openpyxl')
            st.write(dp2)

            x3 = np.array(dp2[dp2.columns[0]], float)
            y3 = np.array(dp2[dp2.columns[1]], float)

            def fungsi_eksponensial(x, a, b, c):
                return a * b ** x + c

            params, _ = curve_fit(fungsi_eksponensial, x3, y3)
            
            a = params[0]
            b = params[1]
            c = params[2]

            yf = fungsi_eksponensial(x3, a, b, c)

            equation = f'y = {a:.2f} * {b:.2f} ** x + {c:.2f}'
            st.write('Persamaan eksponensial:')
            st.write(equation)

            p2 = figure(
                title='Regresi Eksponensial',
                x_axis_label=dp2.columns[0],
                y_axis_label=dp2.columns[1])
    
            p2.scatter(x3, y3, marker='circle', size=6, fill_color='blue')
            p2.line(x3, y3, legend_label='Data', line_width=2)
            p2.line(x3, yf, legend_label='Fungsi Eksponensial', line_width=2, line_color='red')

            st.bokeh_chart(p2, use_container_width=True)








