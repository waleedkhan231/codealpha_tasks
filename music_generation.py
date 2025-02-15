import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from music21 import converter, instrument, note, chord, stream
import os

# Load and preprocess MIDI files
def load_midi_files(directory):
    notes = []
    for file in os.listdir(directory):
        if file.endswith(".mid"):
            midi = converter.parse(os.path.join(directory, file))
            notes_to_parse = None

            try:
                parts = instrument.partitionByInstrument(midi)
                if parts:
                    notes_to_parse = parts.parts[0].recurse()
                else:
                    notes_to_parse = midi.flat.notes
            except:
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    return notes

# Prepare sequences
def prepare_sequences(notes, sequence_length):
    pitch_names = sorted(set(notes))
    pitch_to_int = dict((pitch, i) for i, pitch in enumerate(pitch_names))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([pitch_to_int[char] for char in sequence_in])
        network_output.append(pitch_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(len(pitch_names))  # Normalize
    network_output = tf.keras.utils.to_categorical(network_output)

    return network_input, network_output, pitch_to_int

# Build the LSTM model
def build_model(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(512, input_shape=(network_input.shape[1], network_input.shape[2]), return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Train the model
def train_model(model, network_input, network_output):
    model.fit(network_input, network_output, epochs=200, batch_size=128)

# Generate music
def generate_music(model, network_input, pitch_to_int, sequence_length):
    start = np.random.randint(0, len(network_input) - 1)
    int_to_pitch = dict((i, pitch) for pitch, i in pitch_to_int.items())
    pattern = network_input[start]
    prediction_output = []

    for _ in range(500):  # Generate 500 notes
        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_pitch[index]
        prediction_output.append(result)

        pattern = np.append(pattern, index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

# Create MIDI file
def create_midi(prediction_output, filename):
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if '.' in pattern:  # Chord
            notes_in_chord = pattern.split('.')
            chord_notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                chord_notes.append(new_note)
            new_chord = chord.Chord(chord_notes)
            new_chord.quarterLength = 0.5
            output_notes.append(new_chord)
        else:  # Single note
            new_note = note.Note(int(pattern))
            new_note.quarterLength = 0.5
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=filename)

# Main function
if __name__ == "__main__":
    # Load MIDI files
    notes = load_midi_files("midi_files")  # Replace with your MIDI file directory
    sequence_length = 100

    # Prepare sequences
    network_input, network_output, pitch_to_int = prepare_sequences(notes, sequence_length)
    n_vocab = len(set(notes))

    # Build and train the model
    model = build_model(network_input, n_vocab)
    train_model(model, network_input, network_output)

    # Generate music
    prediction_output = generate_music(model, network_input, pitch_to_int, sequence_length)
    create_midi(prediction_output, "generated_music.mid")
    print("Music generation complete! Check 'generated_music.mid' for the output.")