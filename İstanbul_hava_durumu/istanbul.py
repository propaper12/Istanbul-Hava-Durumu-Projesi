import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel, Text
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from PIL import Image, ImageTk

def show_password_window():
    password_window = tk.Tk()
    password_window.title("Giriş Doğrulama")
    password_window.geometry("600x400")

    def resize_image(event=None):
        # Pencere boyutlarını al
        width = password_window.winfo_width()
        height = password_window.winfo_height()

        # Resmi yeniden boyutlandır
        resized_image = background_image.resize((width, height), Image.Resampling.LANCZOS)
        background_photo = ImageTk.PhotoImage(resized_image)

        # Arka planı güncelle
        background_label.config(image=background_photo)
        background_label.image = background_photo

    # Arka plan resmini yükleme
    background_image = Image.open("1.jpg")

    # Arka plan etiketini ekleme
    background_label = tk.Label(password_window)
    background_label.place(relwidth=1, relheight=1)
    password_window.bind("<Configure>", resize_image)

    # Orta konumlandırma için bir çerçeve oluştur
    frame = tk.Frame(password_window, bg='#ffffff')
    frame.place(relx=0.5, rely=0.5, anchor='center')

    # Kullanıcı adı etiketi ve girişi
    tk.Label(frame, text="Kullanıcı Adı:", font=("Helvetica", 12), bg='#ffffff').grid(row=0, column=0, padx=10, pady=5)
    username_entry = tk.Entry(frame)
    username_entry.grid(row=0, column=1, padx=10, pady=5)

    # Şifre etiketi ve girişi
    tk.Label(frame, text="Şifre:", font=("Helvetica", 12), bg='#ffffff').grid(row=1, column=0, padx=10, pady=5)
    password_entry = tk.Entry(frame, show='*')
    password_entry.grid(row=1, column=1, padx=10, pady=5)

    def verify_password():
        username = username_entry.get()
        password = password_entry.get()
        if username == "sifre123" and password == "sifre123":
            password_window.destroy()
            start_main_application()
        else:
            messagebox.showerror("Hata", "Kullanıcı adı veya şifre hatalı, lütfen tekrar deneyin.")

    tk.Button(frame, text="Doğrula", command=verify_password).grid(row=2, columnspan=2, pady=10)

    # İlk resmi ayarla
    resize_image()

    password_window.mainloop()

def start_main_application():
    # Ana uygulamanın başlatılması
    main_window = tk.Tk()
    main_window.title("Ana Uygulama")
    main_window.geometry("400x300")
    tk.Label(main_window, text="Ana Uygulama Başlatıldı!", font=("Helvetica", 14)).pack(pady=20)
    main_window.mainloop()

def start_main_application():
    main_root = tk.Tk()
    main_root.title("Ana Uygulama")
    main_root.geometry("500x400")

    app = WeatherAnalysisApp(main_root)
    main_root.mainloop()


class WeatherAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("İstanbul Hava Durumu Analiz Projesi")
        self.filepath = None
        self.df = None
        self.create_widgets()

    def create_widgets(self):

        self.background_image = Image.open("2.jpg")
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        self.background_label = tk.Label(self.root, image=self.background_photo)
        self.background_label.place(relwidth=1, relheight=1)

        self.load_button = tk.Button(self.root, text="Veriyi Yükle", command=self.load_data)
        self.load_button.pack(pady=10)

        self.plot_button = tk.Button(self.root, text="Veriyi Görselleştir", command=self.open_visualizer_window, state=tk.DISABLED)
        self.plot_button.pack(pady=10)

        self.train_model_button = tk.Button(self.root, text="Modeli Eğit", command=self.train_model, state=tk.DISABLED)
        self.train_model_button.pack(pady=10)

        self.analyze_button = tk.Button(self.root, text="Veriyi Analiz Et", command=self.analyze_data, state=tk.DISABLED)
        self.analyze_button.pack(pady=10)

    def load_data(self):
        self.filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.filepath:
            self.df = pd.read_csv(self.filepath)
            self.df = self.df.dropna()
            self.df.rename(columns={
                'DateTime': 'DateTime',
                'Condition': 'Condition',
                'MaxTemp': 'MaxTemp',
                'MinTemp': 'MinTemp',
                'SunRise': 'SunRise',
                'SunSet': 'SunSet',
                'MoonRise': 'MoonRise',
                'MoonSet': 'MoonSet',
                'AvgWind': 'AvgWind',
                'AvgHumidity': 'AvgHumidity',
                'AvgPressure': 'AvgPressure'
            }, inplace=True, errors='ignore')

            self.df['DateTime'] = pd.to_datetime(self.df['DateTime'], format='%Y-%m-%d', errors='coerce')
            self.df.set_index('DateTime', inplace=True)

            self.plot_button.config(state=tk.NORMAL)
            self.train_model_button.config(state=tk.NORMAL)
            self.analyze_button.config(state=tk.NORMAL)
            messagebox.showinfo("Info", "Veri başarıyla yüklendi!")

    def analyze_data(self):
        if self.df is not None:
            try:
                # Veri analizi penceresi
                analysis_window = Toplevel(self.root)
                analysis_window.title("Veri Analizi")
                analysis_window.geometry("800x600")
                analysis_window.configure(bg='#f0f0f0')  # Arka plan rengi

                # Ana çerçeve
                analysis_frame = tk.Frame(analysis_window, bg='#ffffff', bd=2, relief='sunken')
                analysis_frame.pack(expand=True, fill='both', padx=10, pady=10)

                # Kaydırıcı
                scroll_y = tk.Scrollbar(analysis_frame, orient='vertical')
                scroll_y.pack(side='right', fill='y')

                # Metin widget'ı
                analysis_text = Text(analysis_frame, wrap='word', yscrollcommand=scroll_y.set, bg='#ffffff', fg='#333333', font=('Arial', 12))
                analysis_text.pack(expand=True, fill='both')

                scroll_y.config(command=analysis_text.yview)

                analysis_output = []

                # Başlık ve İçerik
                def add_section_title(title):
                    analysis_output.append(f"\n\n{'=' * len(title)}\n{title}\n{'=' * len(title)}\n")

                add_section_title("İlk 50 Satır")
                analysis_output.append(self.df.head(50).to_string())

                add_section_title("Veri Bilgisi")
                analysis_output.append(str(self.df.info()))

                if '2019-02-09' in self.df.index:
                    add_section_title("2019-02-09 ve Sonrası")
                    analysis_output.append(self.df.loc['2019-02-09':].to_string())
                else:
                    add_section_title("Tarih '2019-02-09' İndeks İçerisinde Bulunamadı")
                    analysis_output.append("Tarih '2019-02-09' indeks içerisinde bulunamadı.")

                add_section_title("Sütun ve Satır Sayıları")
                analysis_output.append(f"Sütun Sayısı: {self.df.shape[1]}")
                analysis_output.append(f"Satır Sayısı: {self.df.shape[0]}")

                add_section_title("Boşluklar ve Kardinalite")
                analysis_output.append(pd.DataFrame({
                    'Sayım': self.df.shape[0],
                    'Boşluklar': self.df.isnull().sum(),
                    'Boşluk Yüzdesi': self.df.isnull().sum() / self.df.shape[0] * 100,
                    'Kardinalite': self.df.nunique()
                }).to_string())

                add_section_title("Son 10 Satır")
                analysis_output.append(self.df.tail(10).to_string())

                add_section_title("Sıcaklık İstatistikleri")
                analysis_output.append(f"Max Sıcaklık: {self.df['MaxTemp'].max()}")
                analysis_output.append(f"Min Sıcaklık: {self.df['MaxTemp'].min()}")
                analysis_output.append(f"Ortalama Max Sıcaklık: {self.df['MaxTemp'].mean()}")
                analysis_output.append(f"Std Max Sıcaklık: {self.df['MaxTemp'].std()}")
                analysis_output.append(f"Toplam Max Sıcaklık: {self.df['MaxTemp'].sum()}")
                analysis_output.append(f"Max Min Sıcaklık: {self.df['MinTemp'].max()}")
                analysis_output.append(f"Min Min Sıcaklık: {self.df['MinTemp'].min()}")
                analysis_output.append(f"Ortalama Min Sıcaklık: {self.df['MinTemp'].mean()}")
                analysis_output.append(f"Std Min Sıcaklık: {self.df['MinTemp'].std()}")
                analysis_output.append(f"Max Nem: {self.df['AvgHumidity'].max()}")
                analysis_output.append(f"Min Nem: {self.df['AvgHumidity'].min()}")
                analysis_output.append(f"Ortalama Nem: {self.df['AvgHumidity'].mean()}")
                analysis_output.append(f"Std Nem: {self.df['AvgHumidity'].std()}")

                add_section_title("32°C Üzeri Sıcaklıklar")
                analysis_output.append(self.df[self.df['MaxTemp'] > 32].to_string())

                add_section_title("Maksimum Sıcaklık Kayıtları")
                analysis_output.append(self.df[self.df['MaxTemp'] == self.df['MaxTemp'].max()].to_string())

                add_section_title("Minimum Sıcaklık Kayıtları")
                analysis_output.append(self.df[self.df['MaxTemp'] == self.df['MaxTemp'].min()].head(50).to_string())

                # NaN Değerlerle Başa Çıkma
                self.df['Condition'].fillna('Bilinmiyor', inplace=True)
                add_section_title("NaN'ler Doldurulmuş Veri Çerçevesi")
                analysis_output.append(self.df.to_string())

                add_section_title("NaN Doldurduktan Sonra Veri Tipleri")
                analysis_output.append(self.df.dtypes.to_string())

                # Durum Sayıları
                condition_counts = self.df["Condition"].value_counts()
                condition_counts_normalized = self.df["Condition"].value_counts(normalize=True)
                add_section_title("Durum Sayıları")
                analysis_output.append(condition_counts.to_string())
                add_section_title("Normalleştirilmiş Durum Sayıları")
                analysis_output.append(condition_counts_normalized.to_string())

                # Duruma Göre Sıralanmış Veri Çerçevesi
                sorted_df = self.df.sort_values(by=['Condition'], ascending=False).head()
                add_section_title("Duruma Göre Sıralanmış Veri Çerçevesi")
                analysis_output.append(sorted_df.to_string())

                # 'Bilinmiyor' Durumu İçin Ortalama Değerler
                condition_1_mean = self.df[self.df["Condition"] == 'Bilinmiyor'].mean()
                condition_1_mean2 = self.df[self.df["Condition"] == 'Bilinmiyor']['MaxTemp'].mean()
                condition_1_mean3 = self.df[(self.df['Condition'] == 'Bilinmiyor') & (self.df['MaxTemp'] == 'Yok')]['MinTemp'].max()

                add_section_title("'Bilinmiyor' Durumu İçin Ortalama Değerler")
                analysis_output.append(condition_1_mean.to_string())
                add_section_title("'Bilinmiyor' Durumu İçin Ortalama Max Sıcaklık")
                analysis_output.append(str(condition_1_mean2))
                add_section_title("'Bilinmiyor' Durumu ve Max Temp 'Yok' İçin Max Min Sıcaklık")
                analysis_output.append(str(condition_1_mean3))

                # Metni Text Widget'a Ekleme
                analysis_text.insert('1.0', '\n'.join(analysis_output))

                # Güncellemeleri yap
                analysis_text.update_idletasks()

                print("Veri Analizi Penceresi Oluşturuldu ve İçerik Eklendi.")

            except Exception as e:
                messagebox.showerror("Hata", f"Veriyi analiz ederken bir hata oluştu: {e}")
        else:
            messagebox.showwarning("Uyarı", "Veri yüklenmedi!")

    def open_visualizer_window(self):
        if self.df is not None:
            visualizer_window = Toplevel(self.root)
            visualizer_window.title("Veri Görselleştirme")
            visualizer_window.geometry("800x600")
            app = DataVisualizer(visualizer_window, self.df)
        else:
            messagebox.showwarning("Uyarı", "Veri yüklenmedi!")

    def train_model(self):
        if self.df is not None:
            try:
                # Create a new window for model training results
                model_window = Toplevel(self.root)
                model_window.title("Model Eğitimi")
                model_window.geometry("800x600")

                model_text = Text(model_window, wrap='word')
                model_text.pack(side=tk.LEFT, expand=True, fill='both', padx=10, pady=10)

                # Create a figure canvas for matplotlib
                fig = plt.Figure(figsize=(14, 6), dpi=100)
                canvas = FigureCanvasTkAgg(fig, master=model_window)
                canvas.draw()
                canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

                # Set up the subplots
                ax1 = fig.add_subplot(121)
                ax2 = fig.add_subplot(122)

                feature_columns = ['MaxTemp', 'MinTemp', 'AvgWind', 'AvgHumidity', 'AvgPressure']
                X = self.df[feature_columns]
                y = self.df['MaxTemp']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Model selection
                models = {
                    'Linear Regression': LinearRegression(),
                    'Ridge Regression': Ridge(alpha=1.0),
                    'Lasso Regression': Lasso(alpha=0.1)
                }

                results = []
                for model_name, model in models.items():
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)

                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    results.append((model_name, mse, r2))

                    # Plotting model predictions
                    ax1.scatter(y_test, y_pred)
                    ax1.set_xlabel('Gerçek MaxTemp')
                    ax1.set_ylabel('Tahmin Edilen MaxTemp')
                    ax1.set_title(f'{model_name} - Gerçek vs Tahmin')

                    ax2.hist(y_test - y_pred, bins=30, edgecolor='k')
                    ax2.set_xlabel('Hata')
                    ax2.set_title(f'{model_name} - Hata Dağılımı')

                    plt.tight_layout()
                    canvas.draw()

                    # Save the model
                    joblib.dump(model, f"{model_name.replace(' ', '_')}_model.pkl")

                # Display results in the text widget
                results_text = '\n'.join([f"{name}: MSE = {mse:.4f}, R^2 = {r2:.4f}" for name, mse, r2 in results])
                model_text.insert('1.0', f"Model Performansı:\n{results_text}")

            except Exception as e:
                messagebox.showerror("Hata", f"Modeli eğitirken bir hata oluştu: {e}")
        else:
            messagebox.showwarning("Uyarı", "Veri yüklenmedi!")


class DataVisualizer:
    def __init__(self, root, df):
        self.root = root
        self.df = df

        # Create buttons for each plot
        self.create_buttons()

    def create_buttons(self):
        self.background_image = Image.open("3.jpg")
        self.background_photo = ImageTk.PhotoImage(self.background_image)

        self.background_label = tk.Label(self.root, image=self.background_photo)
        self.background_label.place(relwidth=1, relheight=1)

        btn1 = tk.Button(self.root, text="Durum Dağılımı Pasta Grafiği", command=self.plot_pie_chart)
        btn1.pack(pady=10)

        btn2 = tk.Button(self.root, text="Yağış Miktarı Bar Grafiği", command=self.plot_rain_bar_chart)
        btn2.pack(pady=10)

        btn3 = tk.Button(self.root, text="Max Sıcaklık Bar Grafiği", command=self.plot_max_temp_bar_chart)
        btn3.pack(pady=10)

        btn4 = tk.Button(self.root, text="Hava Durumu Sıklığı Bar Grafiği", command=self.plot_condition_frequency)
        btn4.pack(pady=10)

        btn5 = tk.Button(self.root, text="En Popüler 10 Hava Durumu", command=self.plot_top_conditions)
        btn5.pack(pady=10)

        btn6 = tk.Button(self.root, text="Sıcaklıkların Dağılımı", command=self.plot_temperature_distribution)
        btn6.pack(pady=10)

    def create_plot_window(self, title):
        plot_window = Toplevel(self.root)
        plot_window.title(title)
        plot_window.geometry("1200x800")
        fig = plt.Figure(figsize=(12, 10), dpi=100)
        canvas = FigureCanvasTkAgg(fig, master=plot_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        return fig

    def plot_pie_chart(self):
        fig = self.create_plot_window("Durum Dağılımı Pasta Grafiği")
        ax1 = fig.add_subplot(111)
        condition_counts = self.df['Condition'].value_counts()
        top_conditions = condition_counts.head(4)
        other_conditions = condition_counts[4:]
        top_conditions['Others'] = other_conditions.sum()
        ax1.pie(top_conditions, labels=top_conditions.index, autopct='%.1f%%', startangle=140,
                colors=sns.color_palette("pastel", len(top_conditions)))
        ax1.set_title("Hava Durumu Dağılımı")
        plt.tight_layout()

    def plot_rain_bar_chart(self):
        fig = self.create_plot_window("Yağış Miktarı Bar Grafiği")
        ax2 = fig.add_subplot(111)
        condition_rain = self.df.groupby(['Condition'])['Rain'].sum().reset_index()
        sns.barplot(data=condition_rain, x='Condition', y='Rain', ax=ax2)
        ax2.set_title("Durumlara Göre Yağış Miktarı")
        ax2.set_ylabel("Yağış (mm)")
        ax2.set_xlabel("Durum")
        ax2.tick_params(axis='x', rotation=90)
        plt.tight_layout()

    def plot_max_temp_bar_chart(self):
        fig = self.create_plot_window("Max Sıcaklık Bar Grafiği")
        ax3 = fig.add_subplot(111)
        df_filtered = self.df.dropna(subset=['Condition'])
        df_filtered = df_filtered[~df_filtered['Condition'].isin(['Varies', 'varies'])]
        condition_max_temp = df_filtered.groupby(['Condition'])['MaxTemp'].sum().reset_index()
        sns.barplot(data=condition_max_temp, x='Condition', y='MaxTemp', ax=ax3)
        ax3.set_title("Durumlara Göre Max Sıcaklık")
        ax3.set_ylabel("Max Sıcaklık (°C)")
        ax3.set_xlabel("Durum")
        ax3.tick_params(axis='x', rotation=90)
        plt.tight_layout()

    def plot_condition_frequency(self):
        fig = self.create_plot_window("Hava Durumu Sıklığı Bar Grafiği")
        ax4 = fig.add_subplot(111)
        self.df['Condition'].value_counts().plot(kind='bar', ax=ax4)
        ax4.set_title("Hava Durumu Sıklığı")
        plt.tight_layout()

    def plot_top_conditions(self):
        fig = self.create_plot_window("En Popüler 10 Hava Durumu")
        ax5 = fig.add_subplot(111)
        top_conditions = self.df['Condition'].value_counts().head(10)
        sns.barplot(x=top_conditions.index, y=top_conditions.values, ax=ax5, palette='magma')
        ax5.set_title('En Popüler 10 Hava Durumu')
        ax5.set_xlabel('Durum')
        ax5.set_ylabel('Sayı')
        ax5.tick_params(axis='x', rotation=90)
        plt.tight_layout()

    def plot_temperature_distribution(self):
        fig = self.create_plot_window("Sıcaklıkların Dağılımı")
        ax6 = fig.add_subplot(111)
        ax6.hist(self.df['MaxTemp'], bins=30, alpha=0.5, label='Maksimum Sıcaklık')
        ax6.hist(self.df['MinTemp'], bins=30, alpha=0.5, label='Minimum Sıcaklık')
        ax6.set_xlabel('Sıcaklık (°C)')
        ax6.set_ylabel('Frekans')
        ax6.set_title('Sıcaklıkların Dağılımı')
        ax6.legend()
        plt.tight_layout()


# Şifre penceresini göster
show_password_window()
