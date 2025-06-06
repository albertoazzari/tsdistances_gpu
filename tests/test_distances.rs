use csv::ReaderBuilder;
use std::error::Error;
use tsdistances_gpu::{assert_eq_with_tol, cpu, utils, warps::MultiBatchMode, Float};

fn read_csv<T>(file_path: &str) -> Result<Vec<Vec<T>>, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: 'static + Error, // needed to convert parsing error into Box<dyn Error>
{
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_path(file_path)?;

    let mut records = Vec::new();
    for result in reader.records() {
        let record = result?;
        let row: Vec<T> = record
            .iter()
            .map(|s| s.parse::<T>())
            .collect::<Result<Vec<_>, _>>()?;
        records.push(row);
    }
    Ok(records)
}

const WEIGHT_MAX: Float = 1.0;
const TOL: Float = 1e-8;

fn dtw_weights(len: usize, g: Float) -> Vec<Float> {
    let mut weights = vec![0.0; len];
    let half_len = len as Float / 2.0;
    let e = std::f64::consts::E as Float;
    for i in 0..len {
        weights[i] = WEIGHT_MAX / (1.0 + e.powf(-g * (i as Float - half_len)));
    }
    weights
}

#[test]
fn test_device() {
    let (device, _, _, _, _) = utils::get_device();
    println!(
        "Physical device: {:?} \nmax counts: {:?}, \nmax size: {:?}, \n subgroup size: {:?}, \nmax storage buffer range: {:?}",
        device.physical_device().properties().device_name,
        device.physical_device().properties().max_compute_work_group_count,
        device.physical_device().properties().max_compute_work_group_size,
        device.physical_device().properties().max_subgroup_size,
        device.physical_device().properties().max_storage_buffer_range
    );
}

#[test]
pub fn test_erp() {
    let (device, queue, sba, sda, ma) = utils::get_device();

    let ts = read_csv("data/ts.csv").unwrap();
    let erp_ts: Vec<Vec<Float>> = read_csv("results/erp.csv").unwrap();
    let gap_penalty = 1.0;
    let result = cpu::erp::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &ts,
        &ts,
        gap_penalty,
    );
    for i in 0..ts.len() - 1 {
        for j in i + 1..ts.len() {
            assert_eq_with_tol!(result[i][j], erp_ts[i][j], TOL);
        }
    }
}

#[test]
pub fn test_lcss() {
    let (device, queue, sba, sda, ma) = utils::get_device();

    let data = read_csv("data/ts.csv").unwrap();
    let lcss_ts: Vec<Vec<Float>> = read_csv("results/lcss.csv").unwrap();
    let epsilon = 1.0;
    let result = cpu::lcss::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &data,
        &data,
        epsilon,
    );
    for i in 0..data.len() - 1 {
        for j in i + 1..data.len() {
            assert_eq_with_tol!(result[i][j], lcss_ts[i][j], TOL);
        }
    }
}

#[test]
pub fn test_dtw() {
    let data = read_csv("tests/data/ts.csv").unwrap();
    let dtw_ts: Vec<Vec<Float>> = read_csv("tests/results/dtw.csv").unwrap();
    let n_runs = 50;
    let mut avg_time = std::time::Duration::from_secs(0);
    for _ in 0..n_runs {
        let start_time = std::time::Instant::now();
        let (device, queue, sba, sda, ma) = utils::get_device();
        let result = cpu::dtw::<MultiBatchMode>(
            device.clone(),
            queue.clone(),
            sba.clone(),
            sda.clone(),
            ma.clone(),
            &data,
            &data,
        );
        let elapsed_time = start_time.elapsed();
        avg_time += elapsed_time;
        for i in 0..data.len() - 1 {
            for j in i + 1..data.len() {
                assert_eq_with_tol!(result[i][j], dtw_ts[i][j], TOL);
            }
        }
    }
    avg_time /= n_runs;
    println!(
        "Average DTW computation time over {} runs: {:?}",
        n_runs, avg_time
    );
}

#[test]
pub fn test_wdtw() {
    let (device, queue, sba, sda, ma) = utils::get_device();

    let g = 0.05;
    let data = read_csv("data/ts.csv").unwrap();
    let wdtw_ts: Vec<Vec<Float>> = read_csv("results/wdtw.csv").unwrap();
    let result = cpu::wdtw::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &data,
        &data,
        &dtw_weights(data[0].len(), g),
    );
    for i in 0..data.len() - 1 {
        for j in i + 1..data.len() {
            assert_eq_with_tol!(result[i][j], wdtw_ts[i][j], TOL);
        }
    }
}

#[test]
pub fn test_msm() {
    let (device, queue, sba, sda, ma) = utils::get_device();

    let data = read_csv("data/ts.csv").unwrap();
    let msm_ts: Vec<Vec<Float>> = read_csv("results/msm.csv").unwrap();
    let result = cpu::msm::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &data,
        &data,
    );
    for i in 0..data.len() - 1 {
        for j in i + 1..data.len() {
            assert_eq_with_tol!(result[i][j], msm_ts[i][j], TOL);
        }
    }
}

#[test]
pub fn test_twe() {
    let stiffness = 0.001;
    let penalty = 1.0;
    let data = read_csv("data/ts.csv").unwrap();
    let twe_ts: Vec<Vec<Float>> = read_csv("results/twe.csv").unwrap();

    let start_time = std::time::Instant::now();
    let (device, queue, sba, sda, sa) = utils::get_device();

    let result = cpu::twe::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        sa.clone(),
        &data,
        &data,
        stiffness,
        penalty,
    );
    let elapsed_time = start_time.elapsed();

    println!("TWE computation time: {:.4?}", elapsed_time);

    for i in 0..data.len() - 1 {
        for j in i + 1..data.len() {
            assert_eq_with_tol!(result[i][j], twe_ts[i][j], TOL);
        }
    }
}

#[test]
pub fn test_adtw() {
    let (device, queue, sba, sda, ma) = utils::get_device();
    let warp_penalty = 0.1;
    let data = read_csv("data/ts.csv").unwrap();
    let adtw_ts: Vec<Vec<Float>> = read_csv("results/adtw.csv").unwrap();
    let result = cpu::adtw::<MultiBatchMode>(
        device.clone(),
        queue.clone(),
        sba.clone(),
        sda.clone(),
        ma.clone(),
        &data,
        &data,
        warp_penalty,
    );

    for i in 0..data.len() - 1 {
        for j in i + 1..data.len() {
            assert_eq_with_tol!(result[i][j], adtw_ts[i][j], TOL);
        }
    }
}
