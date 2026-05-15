import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


class LendingClubPreprocessor:

    def __init__(self):

        # =================================================
        # categorical columns
        # =================================================
        self.cat_cols = [
            "grade",
            "sub_grade",
            "home_ownership",
            "verification_status",
            "purpose",
            "addr_state",
            "initial_list_status"
        ]

        # =================================================
        # skewed numerical columns
        # =================================================
        self.log_cols = [
            "annual_inc",
            "loan_amnt",
            "installment",
            "revol_bal"
        ]

        # =================================================
        # columns to remove
        # =================================================
        self.drop_cols = [

            # index
            "Unnamed: 0",

            # identifiers
            "id",
            "member_id",
            "url",

            # free text
            "desc",
            "title",
            "emp_title",

            # leakage
            "funded_amnt",
            "funded_amnt_inv",

            "out_prncp",
            "out_prncp_inv",

            "total_pymnt",
            "total_pymnt_inv",

            "total_rec_prncp",
            "total_rec_int",

            "recoveries",
            "collection_recovery_fee",

            "last_pymnt_d",
            "last_credit_pull_d",

            # hardship
            "hardship_flag",
            "hardship_type",
            "hardship_reason",
            "hardship_status",

            "hardship_start_date",
            "hardship_end_date",

            "hardship_amount",
            "hardship_length",

            "hardship_payoff_balance_amount",
            "hardship_last_payment_amount",

            # settlement
            "debt_settlement_flag",
            "debt_settlement_flag_date",

            "settlement_status",
            "settlement_date",

            "settlement_amount",
            "settlement_percentage",
            "settlement_term",
        ]

        self.num_cols = None

        self.scaler = StandardScaler()

        self.label_encoders = {
            col: LabelEncoder()
            for col in self.cat_cols
        }

        self.num_feature_count = 0
        self.cat_feature_count = 0

        self.fitted = False

    # =====================================================
    # CLEAN DATAFRAME
    # =====================================================

    def _clean_dataframe(self, df):

        df = df.copy()

        # -------------------------------------------------
        # keep only binary target rows
        # -------------------------------------------------
        df = df[
            df["loan_status"].isin([
                "Fully Paid",
                "Charged Off"
            ])
        ]

        # -------------------------------------------------
        # int_rate
        # "13.56%" -> 13.56
        # -------------------------------------------------
        if "int_rate" in df.columns:

            df["int_rate"] = (
                df["int_rate"]
                .astype(str)
                .str.replace("%", "", regex=False)
            )

            df["int_rate"] = pd.to_numeric(
                df["int_rate"],
                errors="coerce"
            )

        # -------------------------------------------------
        # term
        # "36 months" -> 36
        # -------------------------------------------------
        if "term" in df.columns:

            extracted = (
                df["term"]
                .astype(str)
                .str.extract(r"(\d+)")
            )

            df["term"] = pd.to_numeric(
                extracted[0],
                errors="coerce"
            )

        # -------------------------------------------------
        # emp_length
        # "10+ years" -> 10
        # -------------------------------------------------
        if "emp_length" in df.columns:

            extracted = (
                df["emp_length"]
                .astype(str)
                .str.extract(r"(\d+)")
            )

            df["emp_length"] = pd.to_numeric(
                extracted[0],
                errors="coerce"
            )

        # -------------------------------------------------
        # date columns
        # -------------------------------------------------
        date_cols = [
            "issue_d",
            "earliest_cr_line"
        ]

        for col in date_cols:

            if col in df.columns:

                dt = pd.to_datetime(
                    df[col],
                    format="%b-%Y",
                    errors="coerce"
                )

                df[col] = (
                    (dt - pd.Timestamp("1970-01-01"))
                    .dt.days
                )

        # -------------------------------------------------
        # remove fully empty columns
        # -------------------------------------------------
        df = df.dropna(
            axis=1,
            how="all"
        )

        # -------------------------------------------------
        # remove selected columns
        # -------------------------------------------------
        existing_drop_cols = [
            c for c in self.drop_cols
            if c in df.columns
        ]

        df = df.drop(
            columns=existing_drop_cols,
            errors="ignore"
        )

        return df

    # =====================================================
    # FIT
    # =====================================================

    def fit(self, df):

        df = self._clean_dataframe(df)

        # -------------------------------------------------
        # numerical columns
        # -------------------------------------------------
        self.num_cols = [
            c for c in df.columns
            if c not in self.cat_cols
            and c != "loan_status"
        ]

        # -------------------------------------------------
        # categorical encoders
        # -------------------------------------------------
        for col in self.cat_cols:

            if col not in df.columns:
                continue

            vals = (
                df[col]
                .fillna("UNK")
                .astype(str)
            )

            unique_vals = vals.unique().tolist()

            if "UNK" not in unique_vals:
                unique_vals.append("UNK")

            self.label_encoders[col].fit(
                unique_vals
            )

        # -------------------------------------------------
        # numerical dataframe
        # -------------------------------------------------
        num_df = df[self.num_cols].copy()

        # convert to numeric
        for col in num_df.columns:

            num_df[col] = pd.to_numeric(
                num_df[col],
                errors="coerce"
            )

        # fill NaN
        num_df = num_df.fillna(
            num_df.median()
        )

        # -------------------------------------------------
        # log transform skewed columns
        # -------------------------------------------------
        for col in self.log_cols:

            if col in num_df.columns:

                num_df[col] = np.log1p(
                    np.maximum(num_df[col], 0)
                )

        # -------------------------------------------------
        # fix inf / overflow
        # -------------------------------------------------
        num_df = num_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        num_df = num_df.clip(
            lower=-1e10,
            upper=1e10
        )

        num_df = num_df.fillna(
            num_df.median()
        )

        # -------------------------------------------------
        # scaler fit
        # -------------------------------------------------
        self.scaler.fit(
            num_df.values
        )

        self.num_feature_count = len(self.num_cols)
        self.cat_feature_count = len(self.cat_cols)

        self.fitted = True

    # =====================================================
    # TRANSFORM
    # =====================================================

    def transform(self, df):

        assert self.fitted, "Call fit first"
    

        # -----------------------------------------
        # clean rows
        # -----------------------------------------
        df = self._clean_dataframe(df)
    
        # -----------------------------------------
        # target
        # -----------------------------------------
        y = None
    
        if "loan_status" in df.columns:
    
            y = (
                df["loan_status"] == "Charged Off"
            ).astype(np.float32).values
    
            df = df.drop(
                columns=["loan_status"],
                errors="ignore"
            )
            
        # -------------------------------------------------
        # categorical features
        # -------------------------------------------------
        cat_features = []

        for col in self.cat_cols:

            if col not in df.columns:
                continue

            vals = (
                df[col]
                .fillna("UNK")
                .astype(str)
            )

            le = self.label_encoders[col]

            vals = vals.map(
                lambda x:
                x if x in le.classes_
                else "UNK"
            )

            encoded = le.transform(vals)

            cat_features.append(
                encoded.reshape(-1, 1)
            )

        cat_features = np.concatenate(
            cat_features,
            axis=1
        )

        # -------------------------------------------------
        # numerical features
        # -------------------------------------------------
        num_df = df[self.num_cols].copy()

        for col in num_df.columns:

            num_df[col] = pd.to_numeric(
                num_df[col],
                errors="coerce"
            )

        #Kitölteni az üres értékeket a mediánnal, hogy a log transzformáció ne dobjon hibát.
        num_df = num_df.fillna(
            num_df.median()
        )

        # -------------------------------------------------
        # log transform
        # -------------------------------------------------
        for col in self.log_cols:

            if col in num_df.columns:
                # A log1p használata lehetővé teszi a 0 értékek kezelését anélkül, hogy hibát dobna.
                num_df[col] = np.log1p(
                    np.maximum(num_df[col], 0)
                )

        # -------------------------------------------------
        # fix inf / overflow
        # -------------------------------------------------
        
        
        num_df = num_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        num_df = num_df.clip(
            lower=-1e10,
            upper=1e10
        )

        num_df = num_df.fillna(
            num_df.median()
        )

        # -------------------------------------------------
        # scale
        # -------------------------------------------------
        num_features = self.scaler.transform(
            num_df.values
        )

        # -------------------------------------------------
        # concat
        # -------------------------------------------------
        X = np.concatenate(
            [num_features, cat_features],
            axis=1
        )

        return X.astype(np.float32), y

    # =====================================================
    # EMBEDDING INFO
    # =====================================================

    def get_embedding_info(self):

        info = {}

        for col in self.cat_cols:

            le = self.label_encoders[col]

            cardinality = len(le.classes_)

            emb_dim = min(
                50,
                (cardinality // 2) + 1
            )

            info[col] = {
                "cardinality": cardinality,
                "embedding_dim": emb_dim
            }

        return info