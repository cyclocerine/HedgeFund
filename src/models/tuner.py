"""
Hyperparameter Tuner Module
=========================

Modul ini berisi kelas HyperparameterTuner untuk
mencari hyperparameter optimal model prediksi.
"""

import keras_tuner as kt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, BatchNormalization
from tensorflow.keras.layers import Input, Bidirectional, Concatenate, GlobalAveragePooling1D, MultiHeadAttention
from tensorflow.keras.layers import LayerNormalization, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import optuna

class HyperparameterTuner:
    def __init__(self, input_shape, tuning_method='hyperband'):
        self.input_shape = input_shape
        self.tuning_method = tuning_method
        
    def build_cnn_lstm_model(self, hp):
        """Membangun model CNN-LSTM dengan hyperparameter yang dapat di-tuning"""
        model = Sequential()
        
        # Tuning jumlah filter dan kernel size untuk layer CNN pertama
        model.add(Conv1D(
            filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Int('conv1_kernel', min_value=2, max_value=5),
            activation='relu',
            input_shape=self.input_shape
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Tuning jumlah filter dan kernel size untuk layer CNN kedua
        model.add(Conv1D(
            filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=32),
            kernel_size=hp.Int('conv2_kernel', min_value=2, max_value=5),
            activation='relu'
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Tuning jumlah unit LSTM
        model.add(LSTM(
            units=hp.Int('lstm1_units', min_value=50, max_value=200, step=50),
            return_sequences=True
        ))
        model.add(Dropout(hp.Float('dropout3', min_value=0.1, max_value=0.5, step=0.1)))
        
        model.add(LSTM(
            units=hp.Int('lstm2_units', min_value=25, max_value=100, step=25),
            return_sequences=False
        ))
        model.add(Dropout(hp.Float('dropout4', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Tuning learning rate
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.add(Dense(25, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        
        return model
        
    def build_bilstm_model(self, hp):
        """Membangun model Bidirectional LSTM dengan hyperparameter yang dapat di-tuning"""
        model = Sequential()
        
        # Tuning jumlah filter dan kernel size untuk layer CNN
        model.add(Conv1D(
            filters=hp.Int('conv_filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Int('conv_kernel', min_value=2, max_value=5),
            activation='relu',
            input_shape=self.input_shape
        ))
        model.add(MaxPooling1D(pool_size=2))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout1', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Tuning jumlah unit BiLSTM
        model.add(Bidirectional(LSTM(
            units=hp.Int('bilstm1_units', min_value=50, max_value=200, step=50),
            return_sequences=True
        )))
        model.add(Dropout(hp.Float('dropout2', min_value=0.1, max_value=0.5, step=0.1)))
        
        model.add(Bidirectional(LSTM(
            units=hp.Int('bilstm2_units', min_value=25, max_value=100, step=25),
            return_sequences=False
        )))
        model.add(Dropout(hp.Float('dropout3', min_value=0.1, max_value=0.5, step=0.1)))
        
        # Tuning learning rate
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        model.add(Dense(25, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        
        return model
        
    def build_transformer_model(self, hp):
        """Membangun model Transformer dengan hyperparameter yang dapat di-tuning"""
        inputs = Input(shape=self.input_shape)
        
        # Tuning parameter preprocessing
        x = Conv1D(
            filters=hp.Int('conv1_filters', min_value=32, max_value=128, step=32),
            kernel_size=hp.Int('conv1_kernel', min_value=2, max_value=5),
            activation='relu'
        )(inputs)
        x = MaxPooling1D(pool_size=2)(x)
        x = Conv1D(
            filters=hp.Int('conv2_filters', min_value=64, max_value=256, step=32),
            kernel_size=hp.Int('conv2_kernel', min_value=2, max_value=5),
            activation='relu'
        )(x)
        x = MaxPooling1D(pool_size=2)(x)
        
        # Tuning parameter transformer
        head_size = hp.Int('head_size', min_value=128, max_value=512, step=128)
        num_heads = hp.Int('num_heads', min_value=2, max_value=8, step=2)
        ff_dim = hp.Int('ff_dim', min_value=2, max_value=8, step=2)
        
        attention_output = MultiHeadAttention(
            key_dim=head_size,
            num_heads=num_heads,
            dropout=hp.Float('attention_dropout', min_value=0.1, max_value=0.5, step=0.1)
        )(x, x)
        x = Add()([attention_output, x])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Tuning learning rate
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        
        x = GlobalAveragePooling1D()(x)
        x = Dense(hp.Int('dense1_units', min_value=32, max_value=128, step=32), activation='relu')(x)
        x = Dropout(hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1))(x)
        x = Dense(hp.Int('dense2_units', min_value=16, max_value=64, step=16), activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mean_squared_error'
        )
        
        return model
        
    def tune_model(self, model_type, X_train, y_train, X_val, y_val, max_trials=10, executions_per_trial=1):
        if self.tuning_method == 'bayesian':
            def objective(trial):
                if model_type == 'cnn_lstm':
                    model = self.build_cnn_lstm_model(trial)
                elif model_type == 'bilstm':
                    model = self.build_bilstm_model(trial)
                elif model_type == 'transformer':
                    model = self.build_transformer_model(trial)
                else:
                    raise ValueError(f"Model type {model_type} not supported for tuning")
                callbacks = [
                    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
                ]
                history = model.fit(
                    X_train, y_train,
                    epochs=50,
                    batch_size=32,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=0
                )
                val_loss = min(history.history['val_loss'])
                return val_loss
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=max_trials)
            best_trial = study.best_trial
            print("\nBest hyperparameters found:")
            for param, value in best_trial.params.items():
                print(f"{param}: {value}")
            # Build model with best params
            if model_type == 'cnn_lstm':
                model = self.build_cnn_lstm_model(optuna.trial.FixedTrial(best_trial.params))
            elif model_type == 'bilstm':
                model = self.build_bilstm_model(optuna.trial.FixedTrial(best_trial.params))
            elif model_type == 'transformer':
                model = self.build_transformer_model(optuna.trial.FixedTrial(best_trial.params))
            else:
                raise ValueError(f"Model type {model_type} not supported for tuning")
            return model
        else:
            if model_type == 'cnn_lstm':
                tuner = kt.Hyperband(
                    self.build_cnn_lstm_model,
                    objective='val_loss',
                    max_epochs=50,
                    factor=3,
                    directory='tuning',
                    project_name='cnn_lstm_tuning'
                )
            elif model_type == 'bilstm':
                tuner = kt.Hyperband(
                    self.build_bilstm_model,
                    objective='val_loss',
                    max_epochs=50,
                    factor=3,
                    directory='tuning',
                    project_name='bilstm_tuning'
                )
            elif model_type == 'transformer':
                tuner = kt.Hyperband(
                    self.build_transformer_model,
                    objective='val_loss',
                    max_epochs=50,
                    factor=3,
                    directory='tuning',
                    project_name='transformer_tuning'
                )
            else:
                raise ValueError(f"Model type {model_type} not supported for tuning")
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            ]
            
            # Mulai pencarian hyperparameter
            print(f"\nStarting hyperparameter tuning for {model_type} model...")
            tuner.search(
                X_train, y_train,
                epochs=50,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=1
            )
            
            # Dapatkan model terbaik
            best_model = tuner.get_best_models(num_models=1)[0]
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            print("\nBest hyperparameters found:")
            for param, value in best_hyperparameters.values.items():
                print(f"{param}: {value}")
            
            return best_model 