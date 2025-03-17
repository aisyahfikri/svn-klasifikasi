import pickle 
from flask import Flask, render_template, request, redirect, url_for, jsonify, flash
import joblib
import mysql.connector
import os
import pandas as pd
from werkzeug.utils import secure_filename
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from nltk.tokenize import word_tokenize
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



app = Flask(__name__)
app.secret_key = "supersecretkey"

def get_db_connection():
    db = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="elsimil"
    )
    return db

# Membuka koneksi dan membuat cursor
db = get_db_connection()
cursor = db.cursor()

UPLOAD_FOLDER = 'uploads'  # Tentukan folder penyimpanan file
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)  # Buat folder jika belum ada

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load Model dan Preprocessing Tools
tfidf_vectorizer = joblib.load("script_deploy/model/tfidf_vectorizer.pkl")
svm_model = joblib.load("script_deploy/model/svm_model.pkl")
label_encoder = joblib.load("script_deploy/model/label_encoder.pkl")
label_mapping = joblib.load("script_deploy/model/label_mapping.pkl")

# Halaman utama
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get_data", methods=["GET"])
def get_data():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    cursor.execute("SELECT COUNT(*) as total FROM ulasan")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT COUNT(*) as positif FROM ulasan WHERE label='positif'")
    positif = cursor.fetchone()["positif"]

    cursor.execute("SELECT COUNT(*) as negatif FROM ulasan WHERE label='negatif'")
    negatif = cursor.fetchone()["negatif"]

    conn.close()

    return jsonify({"total": total, "positif": positif, "negatif": negatif})

# Route untuk menghapus semua data di tabel ulasan
from flask import jsonify

@app.route("/refresh_all_data", methods=["POST"])
def refresh_all_data():
    try:
        # Koneksi ke database
        db = get_db_connection()
        cursor = db.cursor()

        # Menonaktifkan foreign key constraints sementara
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        print("Foreign key checks dinonaktifkan.")

        # Ambil semua tabel dalam database `elsimil`
        cursor.execute("SHOW TABLES;")
        tables = [table[0] for table in cursor.fetchall()]  # Ambil daftar tabel
        print(f"Daftar tabel: {tables}")

        # Urutan tabel berdasarkan dependensi
        # Pastikan tabel anak dihapus sebelum tabel induk
        ordered_tables = ["klasifikasi", "ulasan_preprocessed", "ulasan", "tfidf_result"]

        # Proses setiap tabel berdasarkan urutan dependensi
        for table in ordered_tables:
            if table in tables:
                try:
                    # Hapus semua data dari tabel
                    cursor.execute(f"DELETE FROM `{table}`;")
                    print(f"Semua data pada tabel {table} telah dihapus.")

                    # Reset ID auto_increment ke 1
                    cursor.execute(f"ALTER TABLE `{table}` AUTO_INCREMENT = 1;")
                    print(f"ID auto_increment pada tabel {table} telah direset ke 1.")

                except Exception as e:
                    print(f"Error memproses tabel {table}: {e}")
                    continue  # Lanjutkan ke tabel berikutnya jika ada error pada tabel ini

        # Aktifkan kembali foreign key constraints
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")
        print("Foreign key checks diaktifkan kembali.")

        # Commit perubahan dan tutup koneksi
        db.commit()
        cursor.close()
        db.close()

        return jsonify({"message": "Semua data dalam database elsimil telah di-refresh"}), 200

    except Exception as e:
        # Rollback jika terjadi kesalahan
        if 'db' in locals() and db.open:
            db.rollback()
            print("Rollback berhasil.")
        if 'cursor' in locals() and cursor:
            cursor.close()
        if 'db' in locals() and db.open:
            db.close()
        return jsonify({"error": str(e)}), 500



#ULASAN---->
@app.route('/data-ulasan')
def dataulasan():
    try:
        cursor.execute("SELECT id, data_ulasan, label FROM ulasan")
        data = cursor.fetchall()
        print("Data dari database:", data)  # Debugging
        return render_template('dataulasan.html', data=data)
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return "Terjadi kesalahan dalam mengambil data ulasan", 500


@app.route('/import-data', methods=['POST'])
def import_data():
    if 'file' not in request.files:
        flash('No file selected!')
        return redirect(url_for('dataulasan'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file!')
        return redirect(url_for('dataulasan'))

    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    try:
        print(f"File yang diupload: {file.filename}")  # Debugging

        # **Cek ekstensi file**
        if file.filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(filepath)
        else:
            flash('Format file tidak didukung!')
            return redirect(url_for('dataulasan'))

        # **Cek apakah kolom ada**
        print(f"Kolom dalam file: {df.columns.tolist()}")  # Debugging
        if 'data_ulasan' not in df.columns or 'label' not in df.columns:
            flash('Format kolom file salah! Harus ada: data_ulasan, label.')
            return redirect(url_for('dataulasan'))

        # **Masukkan data ke database**
        cursor = db.cursor()
        for _, row in df.iterrows():
            cursor.execute("INSERT INTO ulasan (data_ulasan, label) VALUES (%s, %s)", 
                           (row['data_ulasan'], row['label']))
        db.commit()
        flash('File berhasil diunggah dan data dimasukkan ke database!')

    except Exception as e:
        flash(f'Error saat proses upload: {e}')
        print(f"Error: {e}")  # Debugging

    return redirect(url_for('dataulasan'))


@app.route('/hapus_ulasan/<int:id>', methods=['POST'])
def delete_ulasan(id):
    try:
        cursor.execute("DELETE FROM ulasan WHERE id = %s", (id,))
        db.commit()
        flash("Ulasan berhasil dihapus!", "success")  # Flash Message
    except Exception as e:
        db.rollback()
        flash(f"Terjadi kesalahan: {e}", "danger")
    
    return redirect(url_for('dataulasan'))  # Redirect kembali ke halaman utama

#PREPROCESSING   
def load_csv(file_path, key_col=None, value_col=None):
    try:
        df = pd.read_csv(file_path)
        df = df.apply(lambda x: x.str.strip().str.lower() if x.dtype == "object" else x)

        if key_col and value_col:
            return dict(zip(df[key_col], df[value_col]))
        return set(df.iloc[:, 0])
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {} if key_col else set()

kamus_dict = load_csv("script_deploy/kamuskatabaku.csv", 'tidak_baku', "kata_baku")

stemmer = StemmerFactory().create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopwords = set(stopword_factory.get_stop_words())

def normalize_text(tokens, kamus):
    return [kamus.get(word, word) for word in tokens]

def preprocess_text(text):
    if not text or not isinstance(text, str) or text.strip() == "":
        return {"cleaning": None, "tokenizing": None, "normalization": None, "filtering": None, "stemming": None}

    text = text.lower().strip()

    cleaning = re.sub(r'http\S+|www\S+', '', text)
    cleaning = re.sub(r'[^a-z\s]', '', cleaning)

    tokenizing = cleaning.split()
    normalization = normalize_text(tokenizing, kamus_dict)
    filtering = [word for word in normalization if word not in stopwords]
    stemming = [stemmer.stem(word) for word in filtering]

    return {
        "cleaning": cleaning,
        "tokenizing": tokenizing,
        "normalization": normalization,
        "filtering": filtering,
        "stemming": stemming
    }

# ✅ Route Halaman Preprocessing
@app.route('/preprocessing')
def preprocessing():
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("""
            SELECT ulasan.id, ulasan.label, ulasan_preprocessed.cleaning, 
                   ulasan_preprocessed.tokenizing, ulasan_preprocessed.normalization, 
                   ulasan_preprocessed.filtering, ulasan_preprocessed.stemming 
            FROM ulasan
            LEFT JOIN ulasan_preprocessed ON ulasan.id = ulasan_preprocessed.ulasan_id
        """)
        data = cursor.fetchall()
    except Exception as e:
        flash(f"Error saat mengambil data: {e}", "danger")
        data = []
    finally:
        cursor.close()
    return render_template("preprocessing.html", data=data)

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    cursor = db.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id, data_ulasan FROM ulasan")
        ulasan_data = cursor.fetchall()

        preprocessed_data = []
        for row in ulasan_data:
            result = preprocess_text(row['data_ulasan'])
            preprocessed_data.append((
                row['id'],
                result['cleaning'],
                ",".join(result['tokenizing']) if result['tokenizing'] else None,
                ",".join(result['normalization']) if result['normalization'] else None,
                ",".join(result['filtering']) if result['filtering'] else None,
                ",".join(result['stemming']) if result['stemming'] else None
            ))

        if preprocessed_data:
            cursor.executemany("""
                INSERT INTO ulasan_preprocessed (ulasan_id, cleaning, tokenizing, normalization, filtering, stemming) 
                VALUES (%s, %s, %s, %s, %s, %s) 
                ON DUPLICATE KEY UPDATE 
                    cleaning=VALUES(cleaning), 
                    tokenizing=VALUES(tokenizing), 
                    normalization=VALUES(normalization), 
                    filtering=VALUES(filtering), 
                    stemming=VALUES(stemming)
            """, preprocessed_data)
            db.commit()

        flash('Preprocessing selesai!', 'success')
    except Exception as e:
        db.rollback()
        flash(f'Error: {e}', 'danger')
    finally:
        cursor.close()

    return redirect(url_for('preprocessing'))

# Halaman pembobotan
# Fungsi untuk menghitung total TF-IDF per dokumen
def compute_tfidf():
    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        # 🔥 Ambil data ulasan yang sudah diproses
        cursor.execute("SELECT ulasan_id, stemming FROM ulasan_preprocessed")
        ulasan_data = cursor.fetchall()

        if not ulasan_data:
            print("Tidak ada data ulasan yang tersedia.")
            return

        doc_ids = [row['ulasan_id'] for row in ulasan_data]
        corpus = [row['stemming'] if row['stemming'] else "" for row in ulasan_data]

        # ✅ **Gunakan model TF-IDF yang sudah dilatih sebelumnya**
        tfidf_vectorizer = joblib.load("script_deploy/model/tfidf_vectorizer.pkl")
        tfidf_matrix = tfidf_vectorizer.transform(corpus)

        tfidf_array = tfidf_matrix.toarray()
        total_tfidf_scores = np.sum(tfidf_array, axis=1)

        # Data yang akan dimasukkan ke database
        tfidf_data = [
            (doc_ids[idx], corpus[idx], total_tfidf_scores[idx])  # Simpan teks stemming
            for idx in range(len(doc_ids))
        ]

        # ❌ Matikan constraint sebelum menghapus data (untuk menghindari FK error)
        cursor.execute("SET FOREIGN_KEY_CHECKS = 0;")
        cursor.execute("DELETE FROM tfidf_results")  # Hapus data lama
        cursor.execute("SET FOREIGN_KEY_CHECKS = 1;")  # ✅ Hidupkan kembali constraint

        # ✅ Masukkan data baru ke `tfidf_results`
        cursor.executemany("""
            INSERT INTO tfidf_results (ulasan_id, term, score) 
            VALUES (%s, %s, %s)
        """, tfidf_data)
        
        db.commit()
        print("Perhitungan TF-IDF berhasil disimpan.")

    except Exception as e:
        print(f"Error saat menghitung TF-IDF: {e}")
    
    finally:
        cursor.close()
        db.close()  # ✅ Pastikan koneksi ditutup setelah selesai

# 📌 Route untuk menghitung TF-IDF
@app.route('/compute_tfidf', methods=['POST'])
def compute_tfidf_route():
    try:
        compute_tfidf()
        flash('Perhitungan TF-IDF selesai!', 'success')
    except Exception as e:
        flash(f'Error: {e}', 'danger')
        print(f"Error: {e}")

    return redirect(url_for('pembobotan'))

# 📌 Route untuk menampilkan hasil pembobotan
@app.route('/pembobotan')
def pembobotan():
    try:
        db = get_db_connection()
        cursor = db.cursor(dictionary=True)

        # 🔥 Ambil `term` dan `score` dari tabel `tfidf_results`
        cursor.execute("SELECT term, score FROM tfidf_results ORDER BY ulasan_id")
        data = cursor.fetchall()
        
        cursor.close()
        db.close()

        return render_template('pembobotan.html', data=data)
    except Exception as e:
        flash(f'Error: {e}', 'danger')
        print(f"Error: {e}")
        return redirect(url_for('index'))
    
#SVMKLASIFIKASI    
@app.route('/klasifikasisvm', methods=['GET', 'POST'])
def klasifikasisvm():
    db = get_db_connection()
    cursor = db.cursor(dictionary=True)

    try:
        print("Fungsi klasifikasisvm dipanggil!")

        if request.method == 'POST':  # Proses hanya saat POST request (tombol klasifikasi diklik)
            if 'klasifikasi_button' in request.form:  # Pastikan tombol klasifikasi ditekan
                print("Tombol klasifikasi ditekan!")
                
                # Ambil data ulasan dari database
                cursor.execute("SELECT id, data_ulasan FROM ulasan")
                ulasan_data = cursor.fetchall()

                if not ulasan_data:
                    flash('Database kosong, tidak bisa melakukan klasifikasi.', 'danger')
                    return render_template('klasifikasisvm.html')

                print(f"Jumlah data ulasan yang akan diklasifikasi: {len(ulasan_data)}")

                # Ekstrak teks ulasan dan ID
                texts = [row['data_ulasan'] for row in ulasan_data]
                ulasan_ids = [row['id'] for row in ulasan_data]

                # Transformasi teks ke fitur TF-IDF
                X_tfidf = tfidf_vectorizer.transform(texts)

                # Prediksi menggunakan model SVM
                y_pred = svm_model.predict(X_tfidf)

                # Konversi prediksi ke label teks
                y_pred_labels = label_encoder.inverse_transform(y_pred)

                # Ambil label asli dari database
                cursor.execute("SELECT id, label FROM ulasan")
                label_data = {row['id']: row['label'] for row in cursor.fetchall()}

                # Mapping label asli sesuai dengan data yang ada
                y_true = [label_data.get(ulasan_id, "Netral") for ulasan_id in ulasan_ids]

                if not y_true:
                    flash('Tidak ada label asli yang ditemukan.', 'danger')
                    return render_template('klasifikasisvm.html')

                # Hanya gunakan label "Positif" dan "Negatif"
                valid_labels = {"Negatif": 0, "Positif": 1}
                filtered_data = [(true, pred) for true, pred in zip(y_true, y_pred_labels) if true in valid_labels and pred in valid_labels]

                if not filtered_data:
                    flash('Tidak ada data yang dapat diklasifikasikan menjadi Positif atau Negatif.', 'danger')
                    return render_template('klasifikasisvm.html')

                # Encode label
                y_true_encoded, y_pred_encoded = zip(*[(valid_labels[t], valid_labels[p]) for t, p in filtered_data])

                # Hitung metrik evaluasi
                acc = float(accuracy_score(y_true_encoded, y_pred_encoded))
                precision = float(precision_score(y_true_encoded, y_pred_encoded, average='binary', zero_division=1))
                recall = float(recall_score(y_true_encoded, y_pred_encoded, average='binary', zero_division=1))
                f1 = float(f1_score(y_true_encoded, y_pred_encoded, average='binary', zero_division=1))

                print(f"Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1-score: {f1}")

                # Hitung Confusion Matrix
                cm = confusion_matrix(y_true_encoded, y_pred_encoded)
                tn, fp, fn, tp = map(int, cm.ravel()) if cm.shape == (2, 2) else (0, 0, 0, 0)

                # Siapkan data untuk batch insert/update
                update_data = []
                insert_data = []
                cursor.execute("SELECT ulasan_id FROM klasifikasi")
                existing_ids = {row['ulasan_id'] for row in cursor.fetchall()}

                for ulasan_id, label in zip(ulasan_ids, y_pred_labels):
                    if label not in valid_labels:
                        continue  # Lewati jika label bukan Positif/Negatif

                    if ulasan_id in existing_ids:
                        update_data.append((label, acc, precision, recall, f1, tp, tn, fp, fn, ulasan_id))
                    else:
                        insert_data.append((ulasan_id, label, acc, precision, recall, f1, tp, tn, fp, fn))

                # Batch update
                if update_data:
                    cursor.executemany("""
                        UPDATE klasifikasi
                        SET label_prediksi = %s, accuracy = %s, precision_value = %s, recall_value = %s, f1_score = %s,
                            true_positive = %s, true_negative = %s, false_positive = %s, false_negative = %s
                        WHERE ulasan_id = %s
                    """, update_data)

                # Batch insert
                if insert_data:
                    cursor.executemany("""
                        INSERT INTO klasifikasi (ulasan_id, label_prediksi, accuracy, precision_value, recall_value, f1_score,
                                                 true_positive, true_negative, false_positive, false_negative)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, insert_data)

                db.commit()
                flash('Klasifikasi berhasil dilakukan.', 'success')

                return render_template('klasifikasisvm.html', accuracy_score=acc, f1_score=f1,
                                       precision_score=precision, recall_score=recall,
                                       true_positive=tp, true_negative=tn,
                                       false_positive=fp, false_negative=fn)

        return render_template('klasifikasisvm.html')

    except Exception as e:
        db.rollback()  # Rollback jika ada error
        flash(f'Error: {e}', 'danger')
        print(f"Error: {e}")
        return render_template('klasifikasisvm.html')

    finally:
        cursor.close()
        db.close()







if __name__ == '__main__':
    app.run(debug=True)  # Debug mode untuk pengembangan
