
import os
import numpy as np
import cv2
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime

class MaskQualityEvaluator:
    def __init__(self, lambda_param=1.0, gamma=1.0, epsilon=1e-6):
        self.lambda_param = lambda_param
        self.gamma = gamma
        self.epsilon = epsilon
        
    def standardize_filename(self, filename):

        base, ext = os.path.splitext(filename)
        parts = base.split('_')
        if len(parts) == 2 and parts[-1].isdigit():
            num = int(parts[-1])
            standardized = f"{parts[0]}_{num:03d}{ext}"
            return standardized
        return filename
    
    def compute_compactness(self, mask):
        """ C_s = L^2/A"""
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                     cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return float('inf')
        
        perimeter = cv2.arcLength(contours[0], closed=True)
        area = np.count_nonzero(mask) 
        return (perimeter ** 2) / (area + self.epsilon)
    
    def compute_edge_quality(self, mask):
        """ Q_e = (1/Ne) * sum(||∇M||)"""
      
        mask_uint8 = mask.astype(np.float32) / 255.0  
        
        grad_x = cv2.Sobel(mask_uint8, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(mask_uint8, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        eroded = cv2.erode(mask_uint8, kernel)
        dilated = cv2.dilate(mask_uint8, kernel)
        edge_mask = (dilated - eroded) > 0
        
        Ne = np.sum(edge_mask)
        if Ne == 0:
            return float('inf')
        
        Qe = np.sum(grad_mag[edge_mask]) / Ne
        return Qe
    
    def compute_area_ratio(self, mask, img_size=None):
        """R_a = A/A_img"""
        mask_area = np.count_nonzero(mask)    
        img_area = mask.shape[0] * mask.shape[1]

        return mask_area / (img_area + self.epsilon)

    def compute_statistics(self, masks):
        """Cs、Ra、Qe: median, mean and variance"""
        Cs_list = []
        Ra_list = []
        Qe_list = []
        
        for mask in masks:
            Cs = self.compute_compactness(mask)
            Ra = self.compute_area_ratio(mask)
            Qe = self.compute_edge_quality(mask)
            
            Cs_list.append(Cs)
            Ra_list.append(Ra)
            Qe_list.append(Qe)
        
        stats_dict = {
            'Cs': {
                'median': float(np.median(Cs_list)),
                'mean': float(np.mean(Cs_list)),
                'variance': float(np.var(Cs_list))
            },
            'Ra': {
                'median': float(np.median(Ra_list)),
                'mean': float(np.mean(Ra_list)),
                'variance': float(np.var(Ra_list))
            },
            'Qe': {
                'median': float(np.median(Qe_list)),
                'mean': float(np.mean(Qe_list)),
                'variance': float(np.var(Qe_list))
            }
        }
        
        return stats_dict

        
    def evaluate_mask(self, mask, img_size, Cs_median, Cs_iqr, Ra_median, Ra_std):
        Cs = self.compute_compactness(mask)
        Ra = self.compute_area_ratio(mask, img_size[0]*img_size[1])
        Qe = self.compute_edge_quality(mask)
        
        return Cs, Ra, Qe
    
    def visualize_and_save(self, mask, fname, output_dir, F, Cs, Ra, Qe):

        plt.figure(figsize=(10, 5), dpi=150)
        
        plt.subplot(1, 2, 1)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask: {fname}\nF={F:.2f}, C_s={Cs:.1f}")
        
        # 计算并绘制边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        eroded = cv2.erode(mask, kernel)
        dilated = cv2.dilate(mask, kernel)
        edges = dilated - eroded
        
        plt.subplot(1, 2, 2)
        plt.imshow(edges, cmap='jet')
        plt.title(f"Edges (Q_e={Qe:.1f})\nR_a={Ra:.4f}")
        
        plt.tight_layout()
        
        # 保存可视化结果
        vis_path = os.path.join(output_dir, f"vis_{os.path.splitext(fname)[0]}.png")
        plt.savefig(vis_path, bbox_inches='tight')
        plt.close()
    
    def process_directory_three_stage(
        self,
        mask_dir,
        output_dir,
        ra_thresh=0.80,
        cs_thresh=50.0,
        qe_thresh=3.47,
        save_visualize=False
    ):
        """
        Three-stage screening with full metrics CSV:
        - Pre-compute and cache Ra, Cs, Qe for each variant (original / inverted)
        - Stage decisions: Ra -> Cs -> Qe
        - Save all final valid masks into Final_valid/mask/
        - Emit two CSV files:
            1) three_stage_screen_log.csv (all samples with metrics and pass flags)
            2) final_valid_list.csv (only final picks with metrics and filenames)
        """
        import pandas as pd

        # Timestamped output root
        timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
        base_out = os.path.join(output_dir, timestamp)
        final_dir = os.path.join(base_out, "Final_valid/mask/")
        final_vis_dir = os.path.join(base_out, "Final_valid/mask_vis/")
        os.makedirs(final_dir, exist_ok=True)
        os.makedirs(final_vis_dir, exist_ok=True)

        # Collect files
        mask_files = sorted(
            [f for f in os.listdir(mask_dir) if f.lower().endswith(".png")],
            key=lambda x: int(x.split("_")[1].split(".")[0])
        )
        print(f"Found {len(mask_files)} mask files")

        # Helpers
        def _read_mask(path):
            m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            m = (m > 127).astype(np.uint8) * 255  # enforce {0,255}
            return m

        def _invert_mask(m):
            return (255 - m).astype(np.uint8)

        # Pre-compute metrics for both variants to avoid recomputation
        metrics_cache = {}  # (fname, is_inv) -> dict(mask, Ra, Cs, Qe)
        for fname in tqdm(mask_files, desc="Precomputing metrics (orig/inv)"):
            path = os.path.join(mask_dir, fname)
            m = _read_mask(path)
            for is_inv, m_var in [(False, m), (True, _invert_mask(m))]:
                Ra = self.compute_area_ratio(m_var)
                Cs = self.compute_compactness(m_var)
                Qe = self.compute_edge_quality(m_var)
                metrics_cache[(fname, is_inv)] = {
                    "mask": m_var,
                    "Ra": float(Ra),
                    "Cs": float(Cs),
                    "Qe": float(Qe)
                }

        # Stage decisions
        stage_records = []
        final_pass = []  # (mask_np, out_name, Ra, Cs, Qe)

        for fname in tqdm(mask_files, desc="Screening"):
            for is_inv in (False, True):
                rec = metrics_cache[(fname, is_inv)]
                Ra, Cs, Qe = rec["Ra"], rec["Cs"], rec["Qe"]

                pass_Ra = (Ra <= ra_thresh)
                pass_Cs = pass_Ra and (Cs <= cs_thresh)
                pass_Qe = pass_Cs and (Qe <= qe_thresh)
                final_selected = bool(pass_Qe)

                stage_records.append({
                    "original_filename": fname,
                    "is_inverted": is_inv,
                    "Ra": Ra,
                    "Cs": Cs,
                    "Qe": Qe,
                    "pass_Ra": pass_Ra,
                    "pass_Cs": pass_Cs,
                    "pass_Qe": pass_Qe,
                    "final_selected": final_selected
                })

                if final_selected:
                    std_name = self.standardize_filename(fname)
                    if is_inv:
                        base, ext = os.path.splitext(std_name)
                        std_name = f"{base}_inv{ext}"
                    final_pass.append((rec["mask"], std_name, Ra, Cs, Qe))

        print(f"Final valid (after Qe): {len(final_pass)}")

        # Save masks (+ optional visualization)
        for m_var, out_name, Ra, Cs, Qe in final_pass:
            out_path = os.path.join(final_dir, out_name)
            cv2.imwrite(out_path, m_var)

            if save_visualize:
                self.visualize_and_save(
                    m_var, out_name, final_vis_dir,
                    F=0.0,  # placeholder, F not used in this pipeline
                    Cs=Cs, Ra=Ra, Qe=Qe
                )

        # CSVs
        df_log = pd.DataFrame(stage_records)
        log_csv = os.path.join(base_out, "three_stage_screen_log.csv")
        df_log.to_csv(log_csv, index=False)
        print(f"Stage log saved: {log_csv}")

        df_final = pd.DataFrame([
            {"final_filename": out_name, "Ra": Ra, "Cs": Cs, "Qe": Qe}
            for _, out_name, Ra, Cs, Qe in final_pass
        ])
        final_csv = os.path.join(base_out, "final_valid_list.csv")
        df_final.to_csv(final_csv, index=False)
        print(f"Final list saved: {final_csv}")

        print(f"Done: final {len(final_pass)} / initial {len(mask_files)*2} (including inversions)")
        return {
            "final_count": len(final_pass),
            "final_dir": final_dir,
            "log_csv": log_csv,
            "final_csv": final_csv
        }


    def save_results(self, results, output_dir):
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, 'mask_quality_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"save result to: {csv_path}")


# 使用示例
if __name__ == "__main__":
    mask_dir = "/media/ders/sda1/XS/SPCAM_FAMS/experiments/predictions/Filter_Formula/mask_ori/mask_255_ori_0/"
    base_output_dir = "/media/ders/sda1/XS/SPCAM_FAMS/experiments/predictions/Filter_Formula/"

    evaluator = MaskQualityEvaluator(lambda_param=1.0, gamma=1.0)

    result = evaluator.process_directory_three_stage(
        mask_dir=mask_dir,
        output_dir=base_output_dir,
        ra_thresh=0.80,
        cs_thresh=50.0,
        qe_thresh=3.47,
        save_visualize=False)  

    print(result)

  
    mask_files = sorted(
        [f for f in os.listdir(mask_dir) if f.endswith('.png')],
        key=lambda x: int(x.split('_')[1].split('.')[0]))
    print(f"find {len(mask_files)} mask fold")
    
    masks = []
    valid_files = []
    img_size = None
    
    for fname in tqdm(mask_files, desc="export masks"):
        mask_path = os.path.join(mask_dir, fname)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255  
        masks.append(mask)
        valid_files.append(fname)
        if img_size is None:
            img_size = mask.shape[:2]
    
    stats = evaluator.compute_statistics(masks)

    print("Cs:")
    print(f"median: {stats['Cs']['median']:.2f}")
    print(f"mean: {stats['Cs']['mean']:.2f}")
    print(f"std: {stats['Cs']['variance']:.2f}\n")

    print("Ra:")
    print(f"median: {stats['Ra']['median']:.4f}")
    print(f"mean: {stats['Ra']['mean']:.4f}")
    print(f"std: {stats['Ra']['variance']:.6f}\n")

    print("Qe:")
    print(f"median: {stats['Qe']['median']:.2f}")
    print(f"mean: {stats['Qe']['mean']:.2f}")
    print(f"std: {stats['Qe']['variance']:.2f}")


