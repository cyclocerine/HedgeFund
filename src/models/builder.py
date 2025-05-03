"""
Model Builder Module
==================

Modul ini berisi kelas ModelBuilder untuk
membuat berbagai arsitektur model deep learning.
"""

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Input, Bidirectional, Concatenate, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization, Add
from tensorflow.keras.optimizers import Adam

class ModelBuilder:
    @staticmethod
    def build_cnn_lstm(input_shape):
        """Membangun model CNN-LSTM"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            BatchNormalization(),
            Dropout(0.2),
            
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            
            Dense(25, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model

    @staticmethod
    def build_bilstm(input_shape):
        """Membangun model Bidirectional LSTM"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=input_shape),
            MaxPooling1D(2),
            BatchNormalization(),
            Dropout(0.2),
            
            Bidirectional(LSTM(100, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=False)),
            Dropout(0.2),
            
            Dense(25, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model

    @staticmethod
    def build_transformer(input_shape):
        """Membangun model Transformer"""
        inputs = Input(shape=input_shape)
        
        # Preprocessing
        x = Conv1D(64, 3, activation='relu')(inputs)
        x = MaxPooling1D(2)(x)
        x = Conv1D(128, 3, activation='relu')(x)
        x = MaxPooling1D(2)(x)
        
        # Transformer block
        attention_output = MultiHeadAttention(
            key_dim=256, num_heads=4, dropout=0.1
        )(x, x)
        x = Add()([attention_output, x])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Output processing
        x = GlobalAveragePooling1D()(x)
        x = Dense(50, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(25, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model

    @staticmethod
    def build_ensemble(input_shape):
        """Membangun model Ensemble"""
        inputs = Input(shape=input_shape)
        
        # CNN branch
        cnn = Conv1D(64, 3, activation='relu')(inputs)
        cnn = MaxPooling1D(2)(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Dropout(0.2)(cnn)
        cnn = Conv1D(128, 3, activation='relu')(cnn)
        cnn = GlobalAveragePooling1D()(cnn)
        
        # LSTM branch
        lstm = Bidirectional(LSTM(100, return_sequences=True))(inputs)
        lstm = Dropout(0.2)(lstm)
        lstm = Bidirectional(LSTM(50, return_sequences=False))(lstm)
        
        # Transformer branch
        trans = Conv1D(64, 3, activation='relu')(inputs)
        trans = MaxPooling1D(2)(trans)
        attention_output = MultiHeadAttention(
            key_dim=256, num_heads=4, dropout=0.1
        )(trans, trans)
        trans = Add()([attention_output, trans])
        trans = LayerNormalization(epsilon=1e-6)(trans)
        trans = GlobalAveragePooling1D()(trans)
        
        # Combine branches
        combined = Concatenate()([cnn, lstm, trans])
        combined = Dense(50, activation='relu')(combined)
        combined = BatchNormalization()(combined)
        combined = Dropout(0.2)(combined)
        combined = Dense(25, activation='relu')(combined)
        outputs = Dense(1)(combined)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
        return model 