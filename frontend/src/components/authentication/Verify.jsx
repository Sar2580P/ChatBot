"use client";
import React, { useState, useContext } from "react";
import classes from "../../styles/Auth.module.css";
import AuthenticationContext from "../../store/authentication/Authentication-context";
import useAuth from "../../hook/useAuth";
import Image from "next/image";
import { RxCross1 } from "react-icons/rx";

const Verify = () => {
  const { Auth } = useAuth();
  const AuthenticationCtx = useContext(AuthenticationContext);
  var phone = AuthenticationCtx.details.phone;
  if (phone == "") phone = "Phone number";
  const [values, setValues] = useState({
    code: "",
    open: false,
    error: "",
  });

  const handleChange = (name) => (event) => {
    setValues({ ...values, [name]: event.target.value });
  };

  const submit = async (e) => {
    e.preventDefault();
    const response = await Auth({ otp: values.code, number: phone }, "verify");
    if (response == "true") {
      setValues({ code: "", open: true });
      AuthenticationCtx.setDetails(phone + "_", "", "", "");
      AuthenticationCtx.onHide("VerifyOpen");
    }
  };

  const hideHandler = () => {
    AuthenticationCtx.onHide("VerifyOpen");
  };

  return (
    <div className={classes.container}>
      <div className={classes.box}>
        <div
          className={classes.close}
          onClick={() => {
            hideHandler();
          }}
        >
          <RxCross1 size={30} />
        </div>
        <div className={classes.part1}>
          <div className={classes.part1_left}>
            <h1>Sign up</h1>
            <p
              onClick={() => {
                AuthenticationCtx.onShow("LogInOpen");
              }}
            >
              <span>or</span> login to your account
            </p>
            <div className={classes.underline}> </div>
          </div>
          <div className={classes.part1_right}>
            <Image src={"/logo.jpg"} width={75} height={75} alt="logo" />
          </div>
        </div>
        <div className={classes.form}>
          <input type="text" placeholder={phone} />
          <input
            type="number"
            placeholder="One time password"
            value={values.code}
            onChange={handleChange("code")}
          />
        </div>
        <div className={classes.continue} onClick={submit}>
          <a>VERIFY OTP</a>
        </div>
        <div className={classes.privacy_policy}>
          By creating an account, I accept the Terms & Conditions & Privacy
          Policy
        </div>
      </div>
    </div>
  );
};

export default Verify;
