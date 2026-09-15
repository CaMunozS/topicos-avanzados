"""Pruebas del notebook. No genera métricas académicas ni entrena en test.

Uso: python tests/validate_tarea3.py --data-root /ruta/a/dataset
Requiere TensorFlow, nbformat, OpenCV y las dependencias del notebook.
"""
import argparse
import ast
import gc
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('TF_NUM_INTRAOP_THREADS', '2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS', '2')

import nbformat
import numpy as np
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    target = repo / 'notebooks/actividad3_Lopez_Munoz.ipynb'
    notebook = nbformat.read(target, as_version=4)
    nbformat.validate(notebook)
    original = json.loads((repo / 'notebooks/Tarea 03.ipynb').read_text())
    originals = {c.metadata.get('tarea3_original_index'): c for c in notebook.cells
                 if 'tarea3_original_index' in c.metadata}
    assert list(originals) == list(range(len(original['cells'])))
    for i, cell in enumerate(original['cells']):
        assert originals[i].cell_type == cell['cell_type']
        if cell['cell_type'] == 'markdown' and i not in {2, 26}:
            assert originals[i].source == ''.join(cell['source']), f'Enunciado alterado: {i}'
    tagged = {}
    for i, cell in enumerate(notebook.cells):
        if cell.cell_type == 'code':
            compile(cell.source, f'cell-{i}', 'exec')
            for tag in cell.metadata.get('tags', []):
                tagged[tag] = cell.source
    namespace = {}
    exec(tagged['configuration'], namespace)
    namespace.update(SAVE_TO_DRIVE=False, USE_MIXED_PRECISION=False,
                     OUTPUT_DIR_OVERRIDE=tempfile.mkdtemp(prefix='tarea3-check-'))
    for tag in ['imports', 'output_setup', 'plot_history', 'dice_functions',
                'iou_functions', 'data_functions']:
        exec(tagged[tag], namespace)
    # Load functions without triggering the dataset download or a full experiment.
    for tag in ['manifest', 'split', 'architecture_training', 'q3_comparison']:
        tree = ast.parse(tagged[tag])
        definitions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
        exec(compile(ast.Module(body=definitions, type_ignores=[]), tag, 'exec'), namespace)
    Metric = namespace['OverlapMetric']
    examples = [
        ([1, 0], [1, 0], 1., 1.),
        ([1, 0], [0, 1], 0., 0.),
        ([0, 0], [0, 0], 1., 1.),
        ([1, 1, 0], [1, 0, 0], .5, 2/3),
    ]
    for truth, pred, expected_iou, expected_dice in examples:
        for kind, expected in [('iou', expected_iou), ('dice', expected_dice)]:
            metric = Metric(kind)
            metric.update_state(tf.constant(truth, tf.float32), tf.constant(pred, tf.float32))
            np.testing.assert_allclose(metric.result(), expected)
            # Serialization must work with the metric's own config.
            reconstructed = Metric.from_config(metric.get_config())
            assert reconstructed.kind == kind
    metric = Metric('iou')
    metric.update_state(tf.constant([1., 1.]), tf.constant([1., 0.]))
    metric.update_state(tf.constant([1.]), tf.constant([1.]))
    np.testing.assert_allclose(metric.result(), 2/3)
    metric.reset_state()
    assert all(float(v) == 0 for v in metric.variables)

    # clear_session resets global state in Keras 3. The model factory must restore
    # the configured precision, and retain a float32 segmentation output.
    namespace['PRECISION_POLICY'] = 'mixed_float16'
    tf.keras.backend.clear_session()
    tiny = namespace['compile_model'](namespace['build_laddernet'](
        stages=2, initial_filters=2, depth=2, image_size=32))
    assert tiny.get_layer('input_projection').dtype_policy.name == 'mixed_float16'
    x = tf.random.uniform((1, 32, 32, 3), seed=84)
    y = tf.cast(x[..., :1] > .7, tf.float32)
    mixed_logs = tiny.train_on_batch(x, y, return_dict=True)
    assert all(np.isfinite(v) for v in mixed_logs.values())
    assert tiny(x, training=False).dtype == tf.float32
    del tiny
    namespace['PRECISION_POLICY'] = 'float32'
    tf.keras.backend.clear_session()

    frame = namespace['build_manifest'](args.data_root)
    train, val, test = namespace['split_by_patient'](frame)
    assert len(frame) == 3929 and frame.patient.nunique() == 110
    assert len(train) + len(val) + len(test) == len(frame)
    sample = train.iloc[:5]
    batches = list(namespace['image_generator'](sample, 2, image_size=32))
    assert [len(x) for x, y in batches] == [2, 2, 1]
    assert all(set(np.unique(y)).issubset({0., 1.}) for x, y in batches)
    # Augmentation must move identical image/mask geometry together.
    mask = np.zeros((32, 32, 1), dtype=np.float32)
    mask[3:17, 4:11] = 1
    image = np.repeat(mask, 3, axis=-1)
    augmented_image, augmented_mask = namespace['augment_pair'](image, mask)
    np.testing.assert_array_equal(augmented_image[..., :1], augmented_mask)
    image, masks = batches[0]
    architectures = [{'stages': 2, 'initial_filters': 15, 'depth': 4},
                     *namespace['Q2_CONFIGS']]
    reports = []
    for architecture in architectures:
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(84)
        model = namespace['build_laddernet'](**architecture, image_size=32)
        stages, depth = architecture['stages'], architecture['depth']
        shared = [layer for layer in model.layers if layer.name.endswith('_shared')]
        assert len(shared) == stages * (2 * depth + 1) + stages - 1
        assert all(len(layer._inbound_nodes) == 2 for layer in shared)
        assert sum('_interstage' in layer.name for layer in model.layers) == (stages - 1) * depth
        assert model.output_shape == (None, 32, 32, 1)
        namespace['compile_model'](model)
        result = model.train_on_batch(image, masks, return_dict=True)
        assert all(np.isfinite(v) for v in result.values())
        before = model(image, training=False).numpy()
        checkpoint = Path(namespace['OUTPUT_ROOT']) / 'roundtrip.weights.h5'
        model.save_weights(checkpoint)
        rebuilt = namespace['build_laddernet'](**architecture, image_size=32)
        rebuilt.load_weights(checkpoint)
        after = rebuilt(image, training=False).numpy()
        np.testing.assert_allclose(before, after, atol=1e-6)
        # Parameter count is independent of spatial resolution.
        configured = namespace['build_laddernet'](**architecture, image_size=512)
        assert configured.output_shape == (None, 512, 512, 1)
        assert model.count_params() == configured.count_params()
        reports.append({**architecture, 'parameters': model.count_params(),
                        'shared_blocks': len(shared), 'finite_train_step': True,
                        'weights_roundtrip': True, 'shape_512': True})
        print('ARCHITECTURE OK', reports[-1], flush=True)
        del model, rebuilt, configured
        gc.collect()
    report = {'scope': 'technical_validation_real_data_no_final_training',
              'tensorflow': tf.__version__, 'keras': tf.keras.__version__,
              'original_cells_preserved': len(originals), 'cells': len(notebook.cells),
              'unique_image_mask_pairs': len(frame), 'patients': int(frame.patient.nunique()),
              'splits': {name: {'images': len(part), 'patients': int(part.patient.nunique())}
                         for name, part in [('train', train), ('val', val), ('test', test)]},
              'metrics_known_cases': True, 'batch_invariance': True,
              'precision_restored_after_clear_session': True,
              'mixed_precision_cpu_smoke': True,
              'augmentation_alignment': True, 'last_batch_included': True,
              'architectures': reports}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
