from typing import List

import numpy as np
import torch
from tqdm import tqdm

from chemprop.data import MoleculeDataLoader, MoleculeDataset, StandardScaler
from chemprop.models import MoleculeModel


def predict(model: MoleculeModel,
            data_loader: MoleculeDataLoader,
            disable_progress_bar: bool = False,
            scaler: StandardScaler = None,
            atom_bond_scalers: List[StandardScaler] = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A :class:`~chemprop.models.model.MoleculeModel`.
    :param data_loader: A :class:`~chemprop.data.data.MoleculeDataLoader`.
    :param disable_progress_bar: Whether to disable the progress bar.
    :param scaler: A :class:`~chemprop.features.scaler.StandardScaler` object fit on the training targets.
    :param atom_bond_scalers: A list of :class:`~chemprop.data.scaler.StandardScaler` fitted on each atomic/bond target.
    :return: A list of lists of predictions. The outer list is molecules while the inner list is tasks.
    """
    model.eval()

    preds = []

    for batch in tqdm(data_loader, disable=disable_progress_bar, leave=False):
        # Prepare batch
        batch: MoleculeDataset
        mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch = \
            batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_features()

        if model.is_atom_bond_targets:
            natoms, nbonds = batch.number_of_atoms, batch.number_of_bonds
            constraints_batch = []
            ind = 0
            for i in range(len(model.atom_targets)):
                if model.atom_constraints is None:
                    constraints_batch.append(None)
                elif i < len(model.atom_constraints):
                    mean, std = atom_bond_scalers[ind].means[0], atom_bond_scalers[ind].stds[0]
                    constraints = torch.tensor([(model.atom_constraints[i] - natom * mean) / std for natom in natoms])
                    constraints_batch.append(constraints.to(model.device))
                else:
                    constraints_batch.append(None)
                ind += 1
            for i in range(len(model.bond_targets)):
                if model.bond_constraints is None:
                    constraints_batch.append(None)
                elif i < len(model.bond_constraints):
                    mean, std = atom_bond_scalers[ind].means[0], atom_bond_scalers[ind].stds[0]
                    constraints = torch.tensor([(model.bond_constraints[i] - nbond * mean) / std for nbond in nbonds])
                    constraints_batch.append(constraints.to(model.device))
                else:
                    constraints_batch.append(None)
                ind += 1

        # Make predictions
        with torch.no_grad():
            batch_preds = model(mol_batch, features_batch, atom_descriptors_batch,
                                atom_features_batch, bond_features_batch, constraints_batch)

        if model.is_atom_bond_targets:
            batch_preds = [x.data.cpu().numpy() for x in batch_preds]

            # Inverse scale for each atom/bond target if regression
            if atom_bond_scalers is not None:
                for i, pred in enumerate(batch_preds):
                    batch_preds[i] = atom_bond_scalers[i].inverse_transform(pred)

            # Collect vectors
            preds.append(batch_preds)
        else:
            batch_preds = batch_preds.data.cpu().numpy()

            # Inverse scale if regression
            if scaler is not None:
                batch_preds = scaler.inverse_transform(batch_preds)

            # Collect vectors
            batch_preds = batch_preds.tolist()
            preds.extend(batch_preds)

    if model.is_atom_bond_targets:
        preds = [np.concatenate(x) for x in zip(*preds)]

    return preds
