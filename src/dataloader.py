class SatelliteFireDataset(Dataset):
    NODATA_FILL = {
        0: 200.0, 1: 0.0, 2: 0.0,
        3: 20.0,  4: 0.0, 5: 0.0
    }

    def __init__(self, file_paths, crop_size=256, augment=False, mode='train'):
        self.file_paths = file_paths
        self.crop_size = crop_size
        self.augment = augment
        self.mode = mode
        self.cropper = T.CenterCrop(crop_size)

    def __len__(self):
        return len(self.file_paths)

    def _normalize(self, raw):
        data = raw.astype(np.float32).copy()
        for ch, fill in self.NODATA_FILL.items():
            data[ch] = np.where(data[ch] == -9999.0, fill, data[ch])

        data[0] = np.clip(data[0] / 4000.0, 0.0, 1.0)
        data[1] = np.clip(data[1] / 90.0,   0.0, 1.0)
        data[2] = np.clip((data[2] + 1.0) / 2.0, 0.0, 1.0)
        data[3] = np.clip((data[3] + 20.0) / 70.0, 0.0, 1.0)
        data[4] = np.clip(data[4] / 0.6, 0.0, 1.0)

        high_res = data[[0, 1, 2]]
        low_res = data[[3, 4]]
        target = np.clip(data[5], 0.0, 1.0)
        return high_res, low_res, target

    def _apply_ndvi_refinement(self, targ_t, ndvi_channel):
        local_ndvi_avg = TF.gaussian_blur(ndvi_channel.unsqueeze(0), kernel_size=[21, 21], sigma=[5.0, 5.0]).squeeze(0)
        local_ndvi_sq = TF.gaussian_blur((ndvi_channel ** 2).unsqueeze(0), kernel_size=[21, 21], sigma=[5.0, 5.0]).squeeze(0)
        local_ndvi_std = torch.sqrt(torch.clamp(local_ndvi_sq - local_ndvi_avg ** 2, min=0))
        vegetation_destroyed = (ndvi_channel < (local_ndvi_avg - 0.5 * local_ndvi_std))
        fire_mask = (targ_t > 0.1) & vegetation_destroyed
        refined_fire = fire_mask.float()
        if refined_fire.sum() > 10:
            refined_fire = TF.gaussian_blur(refined_fire.unsqueeze(0), kernel_size=[5, 5], sigma=[1.5, 1.5]).squeeze(0)
        elif targ_t.sum() > 0:
            refined_fire = TF.gaussian_blur(targ_t, kernel_size=[7, 7], sigma=[3.0, 3.0]).squeeze(0)
        else:
            refined_fire = targ_t
        return torch.clamp(refined_fire, 0.0, 1.0)

    def __getitem__(self, idx):
        with rasterio.open(self.file_paths[idx]) as src: raw = src.read()
        high_res, low_res, target = self._normalize(raw)
        high_t, low_t, targ_t = torch.from_numpy(high_res).float(), torch.from_numpy(low_res).float(), torch.from_numpy(target).float().unsqueeze(0)
        high_t, low_t, targ_t = self.cropper(high_t), self.cropper(low_t), self.cropper(targ_t)
        if self.mode == 'train' and targ_t.sum() > 0:
            targ_t = self._apply_ndvi_refinement(targ_t, high_t[2])
        if self.augment and self.mode == 'train':
            if torch.rand(1).item() < 0.5: high_t, low_t, targ_t = TF.hflip(high_t), TF.hflip(low_t), TF.hflip(targ_t)
            if torch.rand(1).item() < 0.5: high_t, low_t, targ_t = TF.vflip(high_t), TF.vflip(low_t), TF.vflip(targ_t)
            if torch.rand(1).item() < 0.3:
                k = torch.randint(0, 4, (1,)).item()
                high_t, low_t, targ_t = torch.rot90(high_t, k, [1, 2]), torch.rot90(low_t, k, [1, 2]), torch.rot90(targ_t, k, [1, 2])
            if torch.rand(1).item() < 0.15: high_t[2, :, :] = 0.0
            if torch.rand(1).item() < 0.1: low_t[0, :, :] = 0.0
            if torch.rand(1).item() < 0.1: low_t[1, :, :] = 0.0
        return high_t, low_t, targ_t
